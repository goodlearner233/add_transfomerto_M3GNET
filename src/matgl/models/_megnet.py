"""PyTorch Geometric implementation of MEGNet.

Uses ``edge_index`` / scatter-based message passing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch #导入PyTorch库
from torch import nn #导入PyTorch的神经网络模块

from matgl.config import DEFAULT_ELEMENTS #导入MATGL的默认元素列表
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion, #BondExpansion 会把两个原子之间的距离，展开成一组函数值，作为边特征。
    EdgeSet2Set,#对应论文中把每条边的特征进行聚合，得到一个图级别的边特征向量。
    EmbeddingBlock,#mbeddingBlock 用来生成节点、边、全局状态的初始特征
    MEGNetBlock,#具体实现在layer中graph_convolution.py，其中前向传播调用了megnetgraphconv.forward，然后其调用了边，节点，状态的更新
    Set2SetReadOut,#把多个节点的特征进行聚合，得到一个图级别的节点特征向量。
)

from ._core import MatGLModel#导入 MatGL 模型基类，让 MEGNet 可以继承它，里面有集成nn.Module和一些通用功能。

if TYPE_CHECKING:
    from typing import Any

    from matgl.graph._converters import GraphConverter

logger = logging.getLogger(__name__)


class MEGNet(MatGLModel):#MATGL继承nn.Module，MEGNet继承MatGLModel，也就是一个nn.Module，里面有一些MATGL特有的功能，比如保存超参数等。
    """PyG implementation of MEGNet."""

    __version__ = 1

    def __init__(
        self,
        dim_node_embedding: int = 16,#节点，也就是原子的 embedding 维度，默认 16
        dim_edge_embedding: int = 100,#边，也就是 bond distance 展开后的维度，默认 100
        dim_state_embedding: int = 2,# global state 的维度，默认 2
        ntypes_state: int | None = None,
        nblocks: int = 3, #MEGNetBlock 的数量，默认 3
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),#也就是把初始的 node / edge / state embedding 先投影到统一维度
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),#MEGNetBlock 里面更新 edge/node/state 用的 MLP 尺寸
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),#这是 readout 之后的最终输出 MLP。
        nlayers_set2set: int = 1,# 一层set2set的 LSTM，轻量，默认 LSTM是一种聚合方式具体原理不用管，知道作用是啥就行
        niters_set2set: int = 2,# set2set 迭代次数，默认 2，迭代越多，聚合的效果越好，但计算量也越大。
        activation_type: str = "softplus2",#激活函数类型，默认 "softplus2"，这是一个平滑的 ReLU 变体，适合化学数据。
        is_classification: bool = False,#表示不是分类任务
        include_state: bool = True,#是否包含全局状态特征，如果 True，就把 state_attr 也输入到 embedding block 里，生成 state embedding；如果 False，就不生成 state embedding，state_feat 会是一个全零的向量。
        dropout: float = 0.0, #dropout 比例，默认 0.0，表示不使用 dropout，增加 dropout 可以防止过拟合，但也可能降低性能。
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: BondExpansion | None = None,# bond_expansion 用来把 bond distance 展开成边特征，如果不传，就用默认的 Gaussian RBF 展开，维度是 dim_edge_embedding，
        cutoff: float = 4.0,#cutoff = 截断半径，默认 4 Å，建图时只考虑距离小于 cutoff 的原子对。
        gauss_width: float = 0.5,#BondExpansion 里面高斯函数的宽度。
        **kwargs,
    ):
        """Initialize MEGNet."""
        super().__init__()

        self.save_args(locals(), kwargs)#把 __init__ 里面传进来的超参数保存下来。

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff#保存 cutoff
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=dim_edge_embedding, width=gauss_width
        )#如果没有传入 bond_expansion，就创建一个默认的 Gaussian RBF 展开，中心从 0 到 cutoff+1，向量维度数量是 dim_edge_embedding，宽度是 gauss_width。

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]#节点的维度状态，从初始 embedding 维度开始，经过 hidden_layer_sizes_input 的 MLP 投影。16 -> 64 -> 32
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]#边的维度状态，从 bond_expansion 的输出维度开始，经过 hidden_layer_sizes_input 的 MLP 投影。100 -> 64 -> 32
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]#全局状态的维度状态，从初始 embedding 维度开始，经过 hidden_layer_sizes_input 的 MLP 投影。2 -> 64 -> 32

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None #根据 activation_type 从 ActivationFunction 枚举类中获取对应的名字，再加上value()激活函数实例，如果没有找到，就抛出 ValueError，提示可用的激活函数类型。

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,#边特征维度，也就是 bond_expansion 的输出维度
            dim_node_embedding=dim_node_embedding,#节点embedding 维度
            ntypes_node=len(self.element_types),#模型认识的元素种类数量，作为节点类型的数量
            ntypes_state=ntypes_state, #state的类型数量，当u是离散类别时使用
            include_state=include_state,#是否包含全局状态特征，如果 True，就把 state_attr 也输入到 embedding block 里，生成 state embedding；如果 False，就不生成 state embedding，state_feat 会是一个全零的向量。
            dim_state_embedding=dim_state_embedding,#全局状态 embedding 维度，如果 include_state 是 True，就生成这个维度的 state embedding；如果 include_state 是 False，这个维度也没什么用，因为 state_feat 会是全零的。
            activation=activation,
        )#把原始输入变成初始特征的模块，节点特征维度是 dim_node_embedding，边特征维度是 dim_edge_embedding，全局状态特征维度是 dim_state_embedding，激活函数是 activation。

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)
         #调用 MLP 把初始的 edge/node/state embedding 投影到统一的维度空间，各自的输入->64->32
         dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]#block后输出的维度是 hidden_layer_sizes_conv 的最后一个元素，也就是 32
        block_args = {
            "conv_hiddens": list(hidden_layer_sizes_conv),#MEGNetBlock 内部 MLP 的隐藏层结构
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }#给每个block的公共参数
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [  # type: ignore[arg-type]
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)  # type: ignore[arg-type]
            for _ in range(nblocks - 1)
        ]#这里创建MEGNETTBLOCK列表，创建多个
        self.blocks = nn.ModuleList(blocks)#就是把多个 MEGNetBlock 正式挂到模型上。之前是列表里面包含了模型，这么写告诉torch不是普通的列表，而是一个包含模型的 ModuleList，这样在训练时才能正确地更新参数。

        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}#准备 set2set 的参数，迭代次数和层数
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)#边特征的 Set2Set readout。
        self.node_s2s = Set2SetReadOut(dim_blocks_out, **s2s_kwargs)  # type: ignore[arg-type]#节点特征的 Set2Set readout。

        self.output_proj = MLP(
            # 2*S2S(out=2*dim) + state -> output
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )#创建最后输出的MLP(set2set拼接后维度会变成原来的两倍)dims是形容MLP的每一层的特征维度的

        self.dropout = nn.Dropout(dropout) if dropout else None #创建dropout层，如果 dropout 比例大于 0，就创建一个 nn.Dropout 实例；如果 dropout 是 0，就不使用 dropout，设置为 None。

        self.is_classification = is_classification
        self.include_state_embedding = include_state#记录是否包含全局状态 embedding 的标志，后续在 forward 里会根据这个标志决定是否生成 state embedding。

    def forward(self, g: Any, state_attr: torch.Tensor | None = None, **kwargs):#主要输入图对象和u
        """Forward pass of MEGNet (PyG).

        Intermediate layer features are stored on ``self.feature_dict`` (keys
        ``edge_attr``, ``embedding``, ``gc_<i>``, ``readout``, ``final``) and
        overwritten on every call.

        Args:
            g: PyG ``Data`` (or ``Data``-like) object with attributes ``node_type``
                (or ``z``), ``pos``, ``edge_index``, and optionally ``pbc_offshift``,
                ``batch`` and ``num_graphs``.
                g里需要有 node_type 或 z（原子类型），pos（原子坐标），edge_index（边列表），可选的 pbc_offshift（周期性边界条件偏移），batch（批次信息）和 num_graphs（图数量）。
            state_attr: Per-graph state attributes, shape ``(num_graphs, ...)``
                (or ``(...,)`` for a single graph).
            **kwargs: Reserved for future extensions.
        """
        #前面很多dim看的眼花了但不重要，其实重点过程原理都可以直接看forward
        node_attr = getattr(g, "node_type", getattr(g, "z", None))#获取节点属性，
        pos = g.pos#获取节点坐标
        edge_index = g.edge_index#获取边列表
        pbc_offshift = getattr(g, "pbc_offshift", None)#周期性的笛卡尔偏移
        batch = getattr(g, "batch", None)# 每个节点属于 batch 里的哪个结构
        num_graphs = getattr(g, "num_graphs", None)#结构的数量
        num_nodes = pos.size(0)#节点数量
        if num_graphs is None:#判断batch里有几个graph
            num_graphs = 1 if batch is None else int(batch.max().item()) + 1

        edge_batch = None if batch is None else batch[edge_index[0]].to(torch.long)#每条边属于 batch 里的哪个结构。

        _, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)#计算边的距离，返回边的向量和距离，这里只用距离来做边特征。
        edge_attr = self.bond_expansion(bond_dist)#把边的距离展开成边特征，得到 edge_attr，维度是 dim_edge_embedding。

        node_feat, edge_feat, state_feat = self.embedding(node_attr, edge_attr, state_attr)#纠正一下embedding是把输入三类变成初始向量
        edge_feat = self.edge_encoder(edge_feat)#这里对应dense操作
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        fea_dict: dict = {
            "edge_attr": edge_attr,
            "embedding": {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat},
        }#保留中间层结果，教程里看过这个过程字典

        # Ensure state_feat has a leading num_graphs dimension for the conv layers.
        if state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0)

        for i, block in enumerate(self.blocks):#遍历所有block
            edge_feat, node_feat, state_feat = block(
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )#把当前的 edge/node/state 特征送进 MEGNetBlock,做一次信息传递，得到更新后的 edge/node/state 特征。实际上是调用src/matgl/layers/_graph_convolution.py里的
            if state_feat.dim() == 1:
                state_feat = state_feat.unsqueeze(0)
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }#gc1, gc2, gc3 保存了每次 block 之后的 edge/node/state 特征。

        node_vec = self.node_s2s(node_feat, batch)#所有节点特征拼接成一个图级别的节点特征向量，维度是 2*dim_blocks_out，因为 set2set 会把每个节点的特征进行聚合，得到一个图级别的特征向量，维度是原来的两倍。
        edge_vec = self.edge_s2s(edge_feat, edge_batch, num_graphs=num_graphs)#对节点和边做set2set readout

        node_vec = node_vec.view(num_graphs, -1)
        edge_vec = edge_vec.view(num_graphs, -1)
        state_vec = state_feat.view(num_graphs, -1)#把 readout 结果整理成 [num_graphs, feature_dim]

        vec = torch.cat([node_vec, edge_vec, state_vec], dim=-1)
        fea_dict["readout"] = vec#这里是把三个 graph-level vector 拼起来。

        if self.dropout:
            vec = self.dropout(vec)

        output = self.output_proj(vec)#把拼接后的图级别特征送进最后的 MLP，得到输出，
        if self.is_classification:
            output = torch.sigmoid(output)

        output = torch.squeeze(output)#去掉多余维度
        fea_dict["final"] = output
        self.feature_dict = fea_dict#把中间特征挂在 self.feature_dict 
        return output#返回预测结果

    def predict_structure(
        self,
        structure,
        state_attr: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ): #直接输入structure的接口
        """Convenience method to directly predict a property from a structure.

        Args:
            structure: Input crystal/molecule.
            state_attr: Graph attributes (optional).
            graph_converter: Custom converter; defaults to ``Structure2Graph``.

        Returns:
            Output property tensor.
        """
        import matgl

        if graph_converter is None:#如果没有传入 graph_converter，就使用默认的 Structure2Graph 来把 structure 转换成图对象。
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, lat, state_attr_default = graph_converter.get_graph(structure)#把 Structure 转成 graph
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])#把晶胞偏移转换成笛卡尔空间偏移。也就是相对坐标乘以晶格向量，得到实际的坐标偏移。
        g.pos = g.frac_coords @ lat[0]#这里是把site坐标(本来是分数坐标转成笛卡尔坐标)
        if state_attr is None:
            state_attr = torch.tensor(state_attr_default, dtype=matgl.float_th)
        return self(g=g, state_attr=state_attr).detach()#这里直接调用 forward 方法，输入图对象和 state_attr，得到预测结果，并用 detach() 把结果从计算图中分离出来，返回一个普通的张量。
#predict_structure() 的完整逻辑
# 如果用户没给 graph_converter
#    就创建 Structure2Graph

# 2. 用 graph_converter.get_graph(structure)
#    把 Structure 转成 PyG graph

# 3. 计算 g.pbc_offshift
#    给周期性边准备真实空间偏移

# 4. 计算 g.pos
#    把分数坐标转成笛卡尔坐标

# 5. 准备 state_attr

# 6. 调用 self(g=g, state_attr=state_attr)
#    也就是调用 forward()

# 7. detach()
#    返回预测结果，但不保留梯度