
"""
Full-dataset training script for Transformer Atomic-Sum M3GNet.

Overview
--------
This script trains the modified M3GNet model on the complete MatPES-PBE-2025.2
dataset. The original M3GNet message-passing and three-body interaction modules
are retained. A Transformer encoder is added after the M3GNet graph-convolution
blocks.

The model follows this pipeline:

    crystal structure
        -> graph construction
        -> distance and three-body basis expansion
        -> M3GNet graph-convolution blocks
        -> final atomic node features
        -> Transformer encoder
        -> gated atomic-energy prediction head
        -> sum atomic energies within each graph
        -> total energy

In mathematical form:

    H = M3GNet(structure)
    H' = TransformerEncoder(H)
    E_i = GatedMLP(h'_i)
    E_total = sum_i E_i

Unlike the earlier Transformer max-pooling implementation, this atomic-sum
readout preserves the extensive form of the total energy.

Data Pipeline
-------------
1. Load the complete MatPES-PBE-2025.2 JSON dataset.
2. Convert each crystal structure into a PyTorch Geometric graph.
3. Associate each graph with its energy, force, and stress labels.
4. Randomly split the dataset using the following ratios:

       training   = 90%
       validation = 5%
       test       = 5%

5. Construct batched data loaders for Lightning training.

This script uses a sample-level random split. Structures derived from the same
parent material may therefore appear in different subsets. A grouped split by
material ID should be used when stricter generalization evaluation is required.

Default Model Configuration
---------------------------
    cutoff radius              = 5.0 Angstrom
    node embedding dimension   = 64
    edge embedding dimension   = 64
    number of M3GNet blocks    = 3
    Transformer attention heads = 4
    Transformer encoder layers  = 1
    Transformer FFN dimension   = 128
    Transformer dropout         = 0.0
    readout                      = atomic-energy sum
    output targets               = 1

Training Configuration
----------------------
    optimizer            = Adam
    initial learning rate = 1e-3
    energy loss weight    = 1.0
    force loss weight     = 1.0
    stress loss weight    = 0.1
    loss function         = Huber loss
    default batch size    = 4
    default epochs        = 30
    numerical precision   = 32-bit
    gradient clipping     = 2.0
    random seed           = 42

Adam adapts the effective update size separately for each model parameter, but
does not change the global learning rate by itself. The MatGL
PotentialLightningModule supplies its default cosine-annealing learning-rate
scheduler unless a custom scheduler is provided.

Energy, Forces, and Stress
--------------------------
M3GNet directly predicts the total energy. The MatGL Potential wrapper computes
forces and stresses by differentiating the predicted energy:

    F_i = -dE_total / dr_i

    stress = dE_total / d(strain)

Force training consequently requires higher-order differentiation through the
Transformer. The training and testing calls are wrapped with:

    sdpa_kernel(SDPBackend.MATH)

This selects the mathematical scaled-dot-product-attention backend, which
supports the higher-order derivatives required for force training.

Checkpoints and Outputs
-----------------------
The script saves:

    best.ckpt
        Checkpoint with the lowest validation total loss.

    last.ckpt
        Checkpoint from the final training epoch.

    metrics.csv
        Per-epoch training and validation metrics.

    run_config.json
        Dataset path, model parameters, and training configuration.

    test_results.json
        Final metrics evaluated using the best checkpoint.

The best checkpoint is automatically loaded for final test-set evaluation.

Single-GPU Example
------------------
    python train_transformer_atomic_sum_full_matpes.py \
        --data /path/to/MatPES-PBE-2025.2.json \
        --output-dir ./runs/full_matpes_transformer \
        --max-epochs 30 \
        --batch-size 4 \
        --num-workers 8 \
        --accelerator gpu \
        --devices 1

Multi-GPU Example
-----------------
    python train_transformer_atomic_sum_full_matpes.py \
        --data /path/to/MatPES-PBE-2025.2.json \
        --output-dir ./runs/full_matpes_transformer \
        --max-epochs 30 \
        --batch-size 4 \
        --num-workers 8 \
        --accelerator gpu \
        --devices 4

When multiple GPUs are requested, Lightning uses distributed data-parallel
training. The batch size is specified per GPU.

Resume Training
---------------
    python train_transformer_atomic_sum_full_matpes.py \
        --data /path/to/MatPES-PBE-2025.2.json \
        --output-dir ./runs/full_matpes_transformer \
        --resume ./runs/full_matpes_transformer/checkpoints/last.ckpt

Notes
-----
- The complete MatPES dataset is large. Initial graph construction may require
  substantial memory, storage space, and preprocessing time.
- Graph caching should be retained so that subsequent runs do not rebuild every
  graph.
- The script prints the imported MatGL source path at startup. Verify that it
  points to the modified local repository rather than an unrelated installed
  MatGL package.
- Batch size and the number of data-loader workers should be adjusted according
  to the available GPU memory, CPU cores, and system memory.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.nn.attention import SDPBackend, sdpa_kernel


# Prefer the modified MatGL source tree beside this script over site-packages.
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matgl  # noqa: E402
from matgl.config import DEFAULT_ELEMENTS  # noqa: E402
from matgl.graph.data import MGLDataLoader, collate_fn_pes, split_dataset  # noqa: E402
from matgl.models._m3gnet import M3GNet  # noqa: E402
from matgl.utils.training import MGLDatasetLoader, PotentialLightningModule, xavier_init  # noqa: E402


def parse_devices(value: str) -> int | str:
    """Convert a numeric device argument to int while preserving Lightning values such as 'auto'."""
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Transformer atomic-sum M3GNet on the full MatPES dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to the complete MatPES JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/transformer_atomic_sum_full_matpes"),
        help="Directory for graph cache, CSV logs, checkpoints, and test results.",
    )
    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=None,
        help="Directory holding a pre-built MGLDataset graph cache (pyg_graph.pt, lattice.pt, "
        "state_attr.pt, labels.json, fingerprint.json) to reuse instead of rebuilding graphs. "
        "Reused only when its fingerprint (cutoff, element ordering) matches this run's config; "
        "otherwise the dataset is rebuilt and re-cached there. Defaults to '<output-dir>/graph_cache'.",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Lightning accelerator ('auto', 'gpu', 'cpu', ...). 'auto' picks GPU when available and "
        "falls back to CPU otherwise, so the default works on CPU-only machines.",
    )
    parser.add_argument("--devices", type=parse_devices, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None, help="Optional Lightning checkpoint to resume from.")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-num-layers", type=int, default=1)
    parser.add_argument("--transformer-dim-ff", type=int, default=128)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    return parser.parse_args()


def convert_stresses_to_matrices(dataset: Any) -> None:
    """Convert MatPES Voigt stresses in-place to the 3x3 tensors expected by PES training."""
    stresses = dataset.labels.get("stresses")
    if stresses is None:
        raise KeyError("The dataset does not contain stress labels.")

    for index, stress in enumerate(stresses):
        array = np.asarray(stress, dtype=float)
        if array.shape == (6,):
            array = voigt_6_to_full_3x3_stress(array)
        elif array.shape != (3, 3):
            raise ValueError(f"Unexpected stress shape at sample {index}: {array.shape}")
        stresses[index] = array.tolist()


def json_ready(value: Any) -> Any:
    """Convert tensors and NumPy values into JSON-serializable Python objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data.is_file():
        raise FileNotFoundError(f"MatPES JSON file not found: {args.data}")
    if args.accelerator in {"gpu", "cuda"} and not torch.cuda.is_available():
        raise RuntimeError("GPU training was requested, but torch.cuda.is_available() is False.")

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("MatGL source:", Path(inspect.getfile(matgl)).resolve())
    print("Dataset:", args.data)
    print("Output directory:", args.output_dir)

    graph_cache_dir = args.graph_cache_dir if args.graph_cache_dir is not None else args.output_dir / "graph_cache"
    print("Graph cache:", graph_cache_dir)

    dataset_start = time.perf_counter()
    dataset = MGLDatasetLoader.from_json(
        args.data,
        cutoff=args.cutoff,
        element_types=DEFAULT_ELEMENTS,
        save_cache=True,
        root=str(graph_cache_dir),
        stress_unit="kbar",
    )
    convert_stresses_to_matrices(dataset)
    print(f"Dataset prepared: {len(dataset):,} structures in {(time.perf_counter() - dataset_start) / 60:.2f} min")

    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.90, 0.05, 0.05],
        shuffle=True,
        random_state=args.seed,
    )
    print("Split sizes:", len(train_data), len(val_data), len(test_data))

    collate = partial(collate_fn_pes, include_stress=True)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate,
        **loader_kwargs,
    )
    print("Batch counts:", len(train_loader), len(val_loader), len(test_loader))

    model = M3GNet(
        element_types=DEFAULT_ELEMENTS,
        is_intensive=False,
        readout_type="transformer",
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dim_ff=args.transformer_dim_ff,
        transformer_dropout=args.transformer_dropout,
    )
    xavier_init(model)

    print("Model:", type(model).__name__)
    print("Final layer:", type(model.final_layer).__name__)
    print("Trainable parameters:", sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

    lit_module = PotentialLightningModule(
        model=model,
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.1,
        loss="huber_loss",
        lr=1e-3,
    )

    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="best-{epoch:03d}-{step}",
        monitor="val_Total_Loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = CSVLogger(save_dir=args.output_dir, name="logs")

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": "32-true",
        "inference_mode": False,
        "logger": logger,
        "callbacks": [checkpoint, LearningRateMonitor(logging_interval="epoch")],
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 100,
        "gradient_clip_val": 2.0,
    }
    if isinstance(args.devices, int) and args.devices > 1:
        trainer_kwargs["strategy"] = "ddp"

    trainer = L.Trainer(**trainer_kwargs)

    config = vars(args).copy()
    config["split"] = [0.90, 0.05, 0.05]
    config["dataset_size"] = len(dataset)
    config["train_size"] = len(train_data)
    config["val_size"] = len(val_data)
    config["test_size"] = len(test_data)
    config["model_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(json_ready(config), indent=2), encoding="utf-8"
    )

    training_start = time.perf_counter()
    # The MATH SDPA backend supports the higher-order derivatives required by force training.
    with sdpa_kernel(SDPBackend.MATH):
        trainer.fit(
            model=lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(args.resume) if args.resume is not None else None,
        )
        test_results = trainer.test(
            model=lit_module,
            dataloaders=test_loader,
            ckpt_path=checkpoint.best_model_path,
        )[0]

    elapsed_minutes = (time.perf_counter() - training_start) / 60
    result = {
        "best_checkpoint": checkpoint.best_model_path,
        "best_val_total_loss": checkpoint.best_model_score,
        "training_and_test_minutes": elapsed_minutes,
        "test": test_results,
    }
    (args.output_dir / "test_results.json").write_text(
        json.dumps(json_ready(result), indent=2), encoding="utf-8"
    )

    print("Training and testing complete.")
    print("Best checkpoint:", checkpoint.best_model_path)
    print("Best validation loss:", checkpoint.best_model_score)
    print("Elapsed minutes:", elapsed_minutes)
    print("Test results:", test_results)


if __name__ == "__main__":
    main()
