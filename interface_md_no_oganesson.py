"""Run constrained relaxation and NVT MD without the Oganesson wrapper.

This script reproduces the visible workflow in Sherif's example:

1. Read a CIF structure.
2. Freeze atoms within a chosen thickness of both Cartesian-z boundaries.
3. Relax atomic positions without changing the simulation cell.
4. Run constant-temperature molecular dynamics.

The force field is the trained M3GNet + global Transformer + Linear Atomic
Sum checkpoint, loaded explicitly rather than through Oganesson.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import matgl
import numpy as np
import torch
from ase.constraints import FixAtoms
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.ase import MolecularDynamics, Relaxer
from matgl.models._m3gnet import M3GNet
from matgl.utils.training import PotentialLightningModule
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relax and simulate the Li-LGPS interface without Oganesson."
    )
    parser.add_argument("--cif", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    # Values explicitly visible in Sherif's script.
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--freeze-size", type=float, default=4.0)
    parser.add_argument("--relax-fmax", type=float, default=0.8)
    parser.add_argument("--md-steps", type=int, default=100_000)
    parser.add_argument("--loginterval", type=int, default=10)

    # Defaults used by Oganesson/ASE, made explicit for reproducibility.
    parser.add_argument("--relax-steps", type=int, default=1000)
    parser.add_argument("--relax-maxstep", type=float, default=0.2)
    parser.add_argument("--ensemble", choices=("nvt", "nve", "nvt_langevin"), default="nvt")
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    return parser.parse_args()


def load_transformer_potential(checkpoint_path: Path, device: torch.device):
    """Rebuild the verified architecture and load its Lightning checkpoint."""
    model = M3GNet(
        element_types=DEFAULT_ELEMENTS,
        is_intensive=False,
        readout_type="transformer",
        transformer_nhead=4,
        transformer_num_layers=1,
        transformer_dim_ff=128,
        transformer_dropout=0.0,
    )

    lightning_module = PotentialLightningModule(
        model=model,
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.1,
        loss="huber_loss",
        lr=1.0e-3,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # strict=True is an architecture safeguard: the run stops if this is not
    # exactly the checkpoint expected by the reconstructed model.
    lightning_module.load_state_dict(checkpoint["state_dict"], strict=True)
    lightning_module.eval()

    potential = lightning_module.model.to(device)
    potential.eval()

    final_layer = type(potential.model.final_layer).__name__
    parameter_count = sum(parameter.numel() for parameter in potential.model.parameters())
    if final_layer != "TransformerAtomicReadOut" or parameter_count != 296_604:
        raise RuntimeError(
            "Unexpected model architecture: "
            f"final_layer={final_layer}, parameters={parameter_count}"
        )

    return potential, checkpoint, final_layer, parameter_count


def find_fixed_z_atoms(atoms, freeze_size: float) -> tuple[list[int], float, float]:
    """Return indices lying within freeze_size of either Cartesian-z end."""
    z_coordinates = atoms.get_positions()[:, 2]
    z_min = float(z_coordinates.min())
    z_max = float(z_coordinates.max())

    fixed_mask = ((z_coordinates - z_min) < freeze_size) | (
        (z_max - z_coordinates) < freeze_size
    )
    fixed_indices = np.flatnonzero(fixed_mask).tolist()

    if not fixed_indices:
        raise RuntimeError("No atoms were selected for the fixed boundary regions.")
    if len(fixed_indices) == len(atoms):
        raise RuntimeError("The selected fixed regions include every atom.")

    return fixed_indices, z_min, z_max


def main() -> None:
    args = parse_args()

    if not args.cif.is_file():
        raise FileNotFoundError(f"CIF not found: {args.cif}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.freeze_size <= 0:
        raise ValueError("--freeze-size must be positive")
    if args.md_steps < 1 or args.relax_steps < 1:
        raise ValueError("Relaxation and MD step counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the full 23,600-atom interface.")
    device = torch.device("cuda:0")

    print("Host:", platform.node(), flush=True)
    print("MatGL source:", matgl.__file__, flush=True)
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("CIF:", args.cif.resolve(), flush=True)
    print("Checkpoint:", args.checkpoint.resolve(), flush=True)
    print("Output directory:", args.output_dir.resolve(), flush=True)

    # Oganesson replacement, part 1: read CIF and convert Structure -> ASE Atoms.
    structure = Structure.from_file(args.cif)
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)

    # Oganesson replacement, part 2: identify and constrain both z boundaries.
    fixed_indices, z_min, z_max = find_fixed_z_atoms(atoms, args.freeze_size)
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    np.savetxt(
        args.output_dir / "fixed_atom_indices.txt",
        np.asarray(fixed_indices, dtype=int),
        fmt="%d",
    )

    print("Atoms:", len(atoms), flush=True)
    print("Composition:", structure.composition, flush=True)
    print("Cartesian z range (A):", (z_min, z_max), flush=True)
    print("Freeze thickness at each z end (A):", args.freeze_size, flush=True)
    print("Number of fixed atoms:", len(fixed_indices), flush=True)
    print("Number of mobile atoms:", len(atoms) - len(fixed_indices), flush=True)

    # Load the user's trained potential explicitly.
    potential, checkpoint, final_layer, parameter_count = load_transformer_potential(
        args.checkpoint, device
    )
    print("Checkpoint epoch:", checkpoint.get("epoch"), flush=True)
    print("Checkpoint global step:", checkpoint.get("global_step"), flush=True)
    print("Final layer:", final_layer, flush=True)
    print("Model parameters:", parameter_count, flush=True)

    config = {
        "model": "M3GNet + global Transformer + Linear Atomic Sum",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "atoms": len(atoms),
        "fixed_atoms": len(fixed_indices),
        "mobile_atoms": len(atoms) - len(fixed_indices),
        "fixed_direction": "Cartesian z",
        "freeze_size_A": args.freeze_size,
        "relax_cell": False,
        "relax_fmax_eV_A": args.relax_fmax,
        "relax_max_steps": args.relax_steps,
        "relax_maxstep_A": args.relax_maxstep,
        "ensemble": args.ensemble,
        "temperature_K": args.temperature,
        "timestep_fs": args.timestep_fs,
        "md_steps": args.md_steps,
        "simulated_time_ps": args.md_steps * args.timestep_fs / 1000.0,
        "loginterval": args.loginterval,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Oganesson replacement, part 3: relax positions while keeping the cell
    # and the selected boundary atoms fixed.
    print("\n=== CONSTRAINED RELAXATION ===", flush=True)
    print("Cell relaxation: False", flush=True)
    print("Target fmax (eV/A):", args.relax_fmax, flush=True)

    relaxer = Relaxer(potential=potential, optimizer="FIRE", relax_cell=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    relaxation_start = perf_counter()

    relaxation = relaxer.relax(
        atoms,
        fmax=args.relax_fmax,
        steps=args.relax_steps,
        traj_file=str(args.output_dir / "relaxation_trajectory.pkl"),
        interval=1,
        verbose=True,
        maxstep=args.relax_maxstep,
    )

    torch.cuda.synchronize(device)
    relaxation_seconds = perf_counter() - relaxation_start
    relaxed_structure = relaxation["final_structure"]
    relaxed_structure.to(filename=args.output_dir / "relaxed_before_md.cif")

    relaxation_trajectory = relaxation["trajectory"]
    relaxed_max_force = float(
        np.linalg.norm(np.asarray(relaxation_trajectory.forces[-1]), axis=1).max()
    )
    print("Relaxation time (s):", relaxation_seconds, flush=True)
    print("Final relaxed maximum force (eV/A):", relaxed_max_force, flush=True)

    # Convert the relaxed structure back to ASE and reapply the same constraint.
    md_atoms = adaptor.get_atoms(relaxed_structure)
    md_atoms.set_constraint(FixAtoms(indices=fixed_indices))

    # Oganesson replacement, part 4: explicit MatGL/ASE molecular dynamics.
    # In this MatGL version ensemble='nvt' means an NVT Berendsen thermostat.
    print("\n=== MOLECULAR DYNAMICS ===", flush=True)
    print("Ensemble:", args.ensemble, flush=True)
    print("Temperature (K):", args.temperature, flush=True)
    print("Timestep (fs):", args.timestep_fs, flush=True)
    print("MD steps:", args.md_steps, flush=True)
    print("Total simulated time (ps):", config["simulated_time_ps"], flush=True)

    md = MolecularDynamics(
        atoms=md_atoms,
        potential=potential,
        ensemble=args.ensemble,
        temperature=args.temperature,
        timestep=args.timestep_fs,
        trajectory=str(args.output_dir / "md_trajectory.traj"),
        logfile=str(args.output_dir / "md.log"),
        loginterval=args.loginterval,
        append_trajectory=False,
    )

    torch.cuda.synchronize(device)
    md_start = perf_counter()
    md.run(args.md_steps)
    torch.cuda.synchronize(device)
    md_seconds = perf_counter() - md_start

    final_structure = adaptor.get_structure(md_atoms)
    final_structure.to(filename=args.output_dir / "final_after_md.cif")

    final_forces = md_atoms.get_forces()
    final_max_force = float(np.linalg.norm(final_forces, axis=1).max())
    summary = {
        **config,
        "relaxation_seconds": relaxation_seconds,
        "relaxed_max_force_eV_A": relaxed_max_force,
        "md_seconds": md_seconds,
        "final_temperature_K": float(md_atoms.get_temperature()),
        "final_potential_energy_eV": float(md_atoms.get_potential_energy()),
        "final_max_force_eV_A": final_max_force,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
        "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== FINISHED ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Relaxed CIF:", (args.output_dir / "relaxed_before_md.cif").resolve(), flush=True)
    print("MD trajectory:", (args.output_dir / "md_trajectory.traj").resolve(), flush=True)
    print("MD log:", (args.output_dir / "md.log").resolve(), flush=True)
    print("Final CIF:", (args.output_dir / "final_after_md.cif").resolve(), flush=True)


if __name__ == "__main__":
    main()
