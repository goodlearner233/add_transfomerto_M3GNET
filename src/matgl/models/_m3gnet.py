"""PyTorch Geometric implementation of M3GNet.

Uses ``edge_index`` / scatter-based message passing and a tensor-bundle
line graph from ``matgl.graph._compute.create_line_graph``.
"""
#1. Structure -> graph
#    由 graph converter 做。
#    得到 node_type / z, edge_index, frac_coords, lattice, pbc_offset 等。
#    在 predict_structure 里还会得到 pos 和 pbc_offshift。

# 2. compute_pair_vector_and_distance
#    根据 pos, edge_index, pbc_offshift
#    计算每条边的 bond_vec 和 bond_dist，也就是 r_ij。

# 3. BondExpansion
#    把 bond_dist 展开成二体径向 basis / expanded_dists。
#    这是普通边特征的距离基展开。

# 4. create_line_graph
#    根据 edge_index 和 bond 信息，
#    找到哪些 bond-bond pair 能组成三体关系。
#    也就是构造 line_edge_index:
#    bond -> bond，对应 j-i-k 角度关系。

# 5. compute_theta_and_phi
#    根据 line_edge_index、bond_vec、bond_dist
#    计算三体角度信息，主要是 cos(theta_jik)。

# 6. SphericalBesselWithHarmonics
#    把三体中的距离和角度展开成 three_body_basis。
#    这是给三体相互作用用的 basis。

# 7. EmbeddingBlock
#    把 node_types、expanded_dists、state_attr
#    变成初始 node_feat、edge_feat、state_feat。

# 8. 多层 M3GNet 循环
#    for i in range(n_blocks):

#        8.1 ThreeBodyInteractions
#            用 three_body_basis 更新 edge_feat，
#            让边特征带上三体/角度环境信息。

#        8.2 M3GNetBlock
#            用 edge_index、edge_feat、node_feat、state_feat、expanded_dists
#            做 graph convolution：
#                edge update
#                node update
#                optional state update

# 9. Readout

#    如果是 interatomic potential / extensive energy:
#        node_feat
#        -> WeightedReadOut
#        -> 每个原子的能量贡献 E_i
#        -> sum_i E_i
#        -> total energy E

#    如果是 general intensive property:
#        node_feat 或 edge_feat
#        -> weighted_atom / reduce_atom / set2set readout
#        -> graph-level vector
#        -> final MLP
#        -> property
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import (
    compute_pair_vector_and_distance,#计算边向量和距离
    compute_theta_and_phi,#计算theta角度和phi，两条边夹角确实只靠 bond_vec 和 bond_dist 就能算出来。代码后面把这个 cos_theta 拿去做角向基展开，给三体 interaction 使用。
    create_line_graph,#把边当成节点，边之间的关系当成边，构建一个 line graph（返回是一个 dict 包含 line graph 的 edge_index 和一些预计算的张量）
    ensure_line_graph_compatibility, #辅助函数确保兼容
)
from matgl.layers import (
    MLP,  # 普通多层感知机，用在输出层、三体 atom update 等位置
    ActivationFunction,  # 根据字符串选择激活函数，比如 swish、softplus2
    BondExpansion,  # 把边距离 r_ij 展开成径向 basis / edge_attr
    EmbeddingBlock,#把原子类型和距离信息编码成特征向量，作为后续 message passing 的输入
    GatedMLP, #gMLP。论文里定义了
    M3GNetBlock,  # M3GNet 的主图卷积层，负责 edge/node/state 消息传递
    Set2SetReadOut,#对MEGNET是核心但对M3GNET只是一个可选参数
    SphericalBesselWithHarmonics,#对应公式2，把三体几何距离+角度变成三体特征向量
    ThreeBodyInteractions,#三体信息聚合回边e_ij
)
from matgl.layers._readout_torch import ReduceReadOut, WeightedAtomReadOut, WeightedReadOut  # Reduce/Weighted readout，用于普通性质或势能分支
from matgl.utils.cutoff import polynomial_cutoff  # 三体 cutoff，让远距离三体作用平滑衰减到 0

from ._core import MatGLModel, _warn_feature_dict_kwarg

if TYPE_CHECKING:
    from matgl.graph._converters import GraphConverter

logger = logging.getLogger(__name__)


class M3GNet(MatGLModel):
    """PyG implementation of the M3GNet model."""

    __version__ = 2

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,  # 默认元素表，决定模型认识哪些原子类型
        dim_node_embedding: int = 64,  # 节点/原子隐藏向量维度，也就是 embedding 之后 node_feat 的维度
        dim_edge_embedding: int = 64,  # 边/bond 隐藏向量维度
        dim_state_embedding: int = 0,  # 全局 state/u 维度，默认 0 表示通常不用 state
        ntypes_state: int | None = None,  # 离散 state 类别数，如果 state 是整数类别才用
        dim_state_feats: int | None = None,  # 连续 state 特征维度，如果 state 是 T/P 等连续值才用
        max_n: int = 3,  # 径向 basis 阶数，控制距离展开大小 对应公式2中z_ln，径向 radial basis 数量，对应球贝塞尔函数里和距离 r 有关的展开通道
        max_l: int = 3,  # 角向 basis 阶数，控制三体角度展开大小，对应公式2中Y_l，角向 angular basis 数量，对应球谐函数里和角度 θ 有关的展开通道
        nblocks: int = 3,  # M3GNet 层数，每层先 three-body 再 graph conv
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",  # 二体距离展开方式，默认球贝塞尔
        is_intensive: bool = True,  # True 做普通性质预测；False 做总能量/势函数 sum
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",  # 普通性质预测时的聚合方式
        task_type: Literal["classification", "regression"] = "regression",  # 任务类型，默认回归
        cutoff: float = 5.0,  # 二体边截断半径
        threebody_cutoff: float = 4.0,  # 三体相互作用截断半径
        units: int = 64,  # block 内部 MLP 隐藏维度
        ntargets: int = 1,  # 输出目标数量，单目标回归就是 1
        use_smooth: bool = False,  # 是否使用 smooth 版本径向 basis
        use_phi: bool = False,  # 是否使用完整方位角 phi，默认主要用 bond-bond 夹角
        niters_set2set: int = 3,  # Set2Set 迭代次数，只在 set2set readout 用
        nlayers_set2set: int = 3,  # Set2Set 内部 LSTM 层数，只在 set2set readout 用
        field: Literal["node_feat", "edge_feat"] = "node_feat",  # readout 聚合 node_feat 还是 edge_feat，默认节点特征
        include_state: bool = False,  # 是否把全局 state/u 加入消息传递
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",  # 激活函数类型
        dropout: float | None = None,  # dropout 比例，默认不用
        **kwargs,
    ):
        """Initialize the M3GNet model."""
        super().__init__()

        self.save_args(locals(), kwargs)  # 保存初始化参数，方便模型保存/加载

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()  # 把字符串激活函数名变成真正的 nn.Module
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS  # 保存元素表，后面 graph converter 和 embedding 都要对齐

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=rbf_type, smooth=use_smooth)  # 创建二体距离展开模块，bond_dist -> expanded_dists；默认用球贝塞尔函数，和 MEGNet 默认 Gaussian 不同

        degree = max_n * max_l * max_l if use_phi else max_n * max_l  # 三体 basis 的维度，use_phi=True 时，三体角度描述更完整，所以 basis 维度更大。
        degree_rbf = max_n if use_smooth else max_n * max_l  # 二体径向 basis 的维度

        self.embedding = EmbeddingBlock(  # 创建初始 embedding 模块，生成 node/edge/state 初始特征
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(  # 创建三体 basis 展开模块，距离+角度 -> three_body_basis
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            use_phi=use_phi,
            use_smooth=use_smooth,
        )
#         这段代码是在创建 nblocks 个“三体边更新器”。
# 每个更新器里有两个网络：
# 1. update_network_atom：用 v_k 生成公式 (2) 里的 sigmoid 权重
# 2. update_network_bond：把 ẽ_ij 转成公式 (3) 里的边修正量
# 真正的 three_body_basis 和 edge_feat 是 forward 里才传进去计算的。
#      
        self.three_body_interactions = nn.ModuleList(  # 每个 block 一个三体更新层，先把角度信息加到 edge_feat;在模型里保存一个三体更新层列表
            [
                ThreeBodyInteractions(
                    update_network_atom=MLP(  # 三体公式里和原子特征有关的更新网络，对应公式2的σ(W_v v_k + b_v)
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),  # 三体公式里更新边特征的 GatedMLP，对应公式3# 对应公式(3)里的 g(W2 ẽ_ij) ⊙ σ(W1 ẽ_ij)，
# GatedMLP 内部有 value/gate 两条不同参数分支，逐元素相乘被封装在 forward 里。
                )
                for _ in range(nblocks)
            ]
        )

        dim_state_feats_used = dim_state_embedding

        self.graph_layers = nn.ModuleList(  # M3GNetBlock 列表，真正做 edge/node/state graph conv
            [
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats_used,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            ]
        )

        if is_intensive:  # 普通 intensive 性质预测分支，输出不随原子数简单相加
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding  # 根据 field 决定用节点特征readout还是用边特征readout
            if readout_type == "set2set":  # 如果选择 Set2Set 聚合，势函数主线一般不走这里
                if field != "node_feat": #只能对节点特征做 Set2Set 聚合，边特征暂时不支持
                    raise NotImplementedError("Set2Set readout on edge features is not implemented for PyG yet.")
                self.readout = Set2SetReadOut(  # type: ignore[call-arg]
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set
                )
                readout_feats = 2 * input_feats + dim_state_feats_used if include_state else 2 * input_feats  #set2set输出维度是 2 * input_feats，如果有 state 就加上 state 的维度
            elif readout_type == "weighted_atom":  # 默认 weighted atom 聚合分支
                self.readout = WeightedAtomReadOut(  # type: ignore[assignment]  #对原子特征做带权重的聚合
                    in_feats=input_feats, dims=[units, units], activation=activation
                )
                readout_feats = units + dim_state_feats_used if include_state else units
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore[assignment] #这里是最简单的readout，求所有节点/边特征求平均
                readout_feats = input_feats + dim_state_feats_used if include_state else input_feats

            dims_final_layer = [readout_feats, units, units, ntargets]  # 最终 MLP 的维度：graph vector -> 输出
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)  # 普通性质预测的输出 MLP
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()  # 分类任务最后接 sigmoid
        else:#如果是 extensive 能量预测分支，输出随原子数简单相加
            if task_type == "classification":#在此情况下，分类任务不适用，因为 extensive 能量预测是回归问题
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(  # type: ignore[assignment]
                in_feats=dim_node_embedding,
                dims=[units, units],
                num_targets=ntargets,
            )

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive
        self.field = field
        self.readout_type = readout_type

    def _readout(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:#根据设置，决定最后 readout 时用 node_feat 还是 edge_feat。
        """Dispatch the configured readout on the right field tensor."""
        x = node_feat if self.field == "node_feat" else edge_feat  # 根据 field 选择读出节点特征或边特征
        if isinstance(self.readout, ReduceReadOut):
            return self.readout(x, batch)
        return self.readout(x, batch)

    def forward(
        self,
        g: Any,
        state_attr: torch.Tensor | None = None,
        l_g: dict[str, torch.Tensor] | None = None,
        return_all_layer_output: bool = False,
    ):
        """Forward pass of M3GNet (PyG).

        Intermediate layer features are always stored on ``self.feature_dict`` after
        every call (overwritten on each forward).

        Args:
            g: PyG ``Data`` (or ``Data``-like) object with attributes
                ``node_type`` (or ``z``), ``pos``, ``edge_index``, optionally
                ``pbc_offshift`` and ``batch`` / ``num_graphs``.
            state_attr: Per-graph state features (optional).
            l_g: Cached line-graph bundle from
                :func:`matgl.graph._compute.create_line_graph`. If ``None``,
                a fresh one is built from ``g``.
            return_all_layer_output: **Deprecated.** Use ``model.feature_dict`` after
                the forward call instead. Will be removed in matgl v5. When ``True``
                the feature dict is still returned for backwards compatibility.
        """
        if return_all_layer_output: 
            _warn_feature_dict_kwarg("return_all_layer_output")
        #读入图信息
        node_types = getattr(g, "node_type", getattr(g, "z", None))  # 读取原子类型，node_type 或 z
        pos = g.pos  # 从图g拿原子笛卡尔坐标
        edge_index = g.edge_index  # 原子图的边连接关系，shape 是 (2, num_edges)
        pbc_offshift = getattr(g, "pbc_offshift", None)  # 周期性边界下的真实空间偏移
        batch = getattr(g, "batch", None)  # batch 中每个节点属于哪个结构
        num_graphs = getattr(g, "num_graphs", None)
        num_nodes = pos.size(0)  # 当前 batch 里的总原子数
        num_bonds = edge_index.size(1)  # 当前 batch 里的总边数
        if num_graphs is None:
            num_graphs = 1 if batch is None else int(batch.max().item()) + 1
        edge_batch = None if batch is None else batch[edge_index[0]].to(torch.long)  # 每条边属于 batch 中哪个结构

# #pos + edge_index
# -> 每条边的向量 bond_vec
# -> 每条边的距离 bond_dist
# -> 距离展开 expanded_dists(下面两句)
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)  # 根据坐标和 edge_index 计算每条边的向量和距离
        expanded_dists = self.bond_expansion(bond_dist)  # 把 r_ij 展开成二体径向 basis e0ij

        if l_g is None:
            l_g = create_line_graph(edge_index, bond_dist, bond_vec, pbc_offshift, num_nodes, self.threebody_cutoff)  # 构造 line graph，找 bond-bond pair 形成三体角度
        else:
            l_g = ensure_line_graph_compatibility(l_g, bond_dist, bond_vec, pbc_offshift, self.threebody_cutoff)  # 复用已有 line graph，只刷新距离和向量

        angles = compute_theta_and_phi(l_g["bond_vec"], l_g["bond_dist"], l_g["line_edge_index"])  # 根据 line_edge_index 计算两条 bond 的夹角 cos(theta)
        three_body_basis = self.basis_expansion(angles["triple_bond_lengths"], angles["cos_theta"], angles["phi"])  # 把三体距离和角度展开成 three_body_basis，即球贝塞尔函数 × 球谐函数得到的三体几何 basis
        three_body_cutoff = polynomial_cutoff(bond_dist, self.threebody_cutoff)  # 三体 cutoff 权重，远距离平滑衰减到 0

        node_feat, edge_feat, state_feat = self.embedding(node_types, expanded_dists, state_attr)  # 生成初始 node/edge/state hidden features；这里把 expanded_dists/e0ij 编码或投影成 edge_feat，不是再做距离展开
        if self.include_state and state_feat is not None and state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0)  # 单图 state 补一个 batch 维度

        fea_dict: dict[str, Any] = {
            "bond_expansion": expanded_dists,
            "three_body_basis": three_body_basis,
            "embedding": {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat},
        }

        edge_dst_atom = edge_index[1]  # 每条边的终点原子 index，三体更新里要用
        line_edge_index = l_g["line_edge_index"]  # 线图边，表示哪些 bond-bond pair 组成三体
        n_triple_ij = l_g["n_triple_ij"]  # 每条 bond 参与的三体数量，用于聚合回 edge_feat

        for i in range(self.n_blocks):  # 逐层执行 M3GNet：每层先三体更新边，再 graph conv
            edge_feat = self.three_body_interactions[i](  # 用三体 basis 更新 edge_feat，把角度信息写进边特征
                edge_dst_atom,
                line_edge_index,
                n_triple_ij,
                num_bonds,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](  # 调用 M3GNetBlock，执行 edge/node/state 消息传递
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                expanded_dists,
                batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }

        if self.is_intensive: #readout输出分两种情况
            field_vec = self._readout(node_feat, edge_feat, batch)  # 普通性质分支：把 node/edge 特征聚合成 graph vector
            if self.include_state and state_feat is not None:
                state_view = state_feat.view(num_graphs, -1)  # 把 state_feat 整理成每个 graph 一行
                readout_vec = torch.hstack([field_vec, state_view])  # 如果使用 state，就把 graph vector 和 state 拼接
            else:
                readout_vec = field_vec
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)  # 普通性质分支：graph vector 经过 MLP 得到输出
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:#如果readout走 extensive 能量预测分支，输出随原子数简单相加
            atomic = self.final_layer(node_feat)  # (num_nodes, ntargets)
            fea_dict["readout"] = atomic
            atomic = atomic.view(-1)
            if batch is None:
                output = atomic.sum().view(1)  # 单个结构：所有原子能量贡献求和得到总能量
            else: #多个结构batch时，用 index_add 按 graph 分组把原子能量求和
                output = torch.zeros(num_graphs, dtype=atomic.dtype, device=atomic.device)
                output = output.index_add(0, batch.to(torch.long), atomic)  # batch 多个结构：按 graph 分组把原子能量求和

        fea_dict["final"] = output
        self.feature_dict = fea_dict  # 保存中间特征，方便调试/查看每层输出
        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)  # 去掉多余维度后返回预测结果，这里是能量

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to predict a property from a structure (PyG).

        Args:
            structure: An input crystal/molecule.
            state_feats: Optional state attributes.
            graph_converter: Graph converter. Defaults to ``Structure2Graph``.
            output_layers: Currently unused; kept for API symmetry with other models.
            return_features: **Deprecated.** Use ``model.feature_dict`` after calling
                ``predict_structure`` instead. Will be removed in matgl v5.
        """
        import matgl

        if return_features:
            _warn_feature_dict_kwarg("return_features")

        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # 默认用 Structure2Graph 把 pymatgen Structure 转成图
        g, lat, state_attr_default = graph_converter.get_graph(structure)  # 得到图 g、晶格 lat 和默认 state
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])  # 把周期性偏移转成笛卡尔空间偏移
        g.pos = g.frac_coords @ lat[0]  # 分数坐标乘晶格矩阵得到笛卡尔坐标
        if state_feats is None:
            state_feats = torch.tensor(state_attr_default, dtype=matgl.float_th)  # 没有手动传 state 时使用 graph converter 给的默认 state
        if return_features:
            self(g=g, state_attr=state_feats)
            return self.feature_dict
        return self(g=g, state_attr=state_feats).detach()  # 直接调用 forward 做预测，并 detach 出普通张量
# predict_structure是一个方便预测的接口，直接输入 pymatgen Structure
# -> 自动转成 graph
# -> 调用 forward()
# -> 返回预测结果
