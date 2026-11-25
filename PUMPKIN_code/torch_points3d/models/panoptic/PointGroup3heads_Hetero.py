"""
PointGroup3heads_Hetero — Multi-view 2D projection + heterogeneous graph fusion.

Extends the standard PointGroup-3heads architecture with:
  - 6-view orthographic projection of RGB point clouds
  - ResNet18 2D CNN for image features
  - 3D superpoint voxelisation
  - Heterogeneous graph message passing (3D superpoints ↔ 2D patches)
  - Fusion of 3D backbone and upsampled superpoint features

The 3D backbone, scorer, and clustering / loss methods are shared with
PointGroup3heads.  Only __init__ and forward are overridden.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch_geometric.data import Data
from torch_scatter import scatter

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from torch_points3d.core.losses import (
    offset_loss, instance_iou_loss, mask_loss,
    instance_ious, discriminative_loss,
)
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.core.data_transform.multiview import (
    multi_view_project,
    pixel_to_patch_map,
    select_views,
    VIEW_DIRS,
    VIEW_ORDER_TOP_FIRST,
    NUM_VIEWS,
)
from .structure_3heads import PanopticLabels, PanopticResults
from .growth_prior import resolve_growth_prior
from .causal_merge import (
    CausalMerge,
    cluster_summaries,
    cluster_majority_labels,
    build_merge_pairs,
    pair_feature_tensor,
    pair_labels_from_gt,
    union_find_merge,
)
from torch_points3d.utils import hdbscan_cluster, meanshift_cluster
from torch_points3d.utils import is_list
from torch_points_kernels import region_grow

import logging
from collections import OrderedDict


# ── ACGO (S3 优化算法臂) ─────────────────────────────────────────────────────
# 任务集（用户 2026-09-24 定 = 3 个）。合并两个 offset 子损失是因为它们共用
# `Offset` head、监督同一物理量（投票向量）的两面；拆开会把"同一目标内部的
# 噪声"算成任务间冲突。
_ACGO_TASKS = ("offset", "embed", "score")

# SHARED = 被 >=2 个任务触达的参数前缀（做冲突检测 + PCGrad 投影）。
# 本 config 下的**实测**归属（probe_grad_conflict.py 2026-09-24 在 S2 checkpoint 上
# 跑了 60 个真实 iteration，`touched_counts` 给出）：
#   proj_gnn_to_fusion / fusion_mlp / backbone_proj / growth_proj /
#   causal_growth_attn → 全部汇入 fused_feat_all ⇒ 3 个任务共同触达 ✓
#   scorer_fusion_proj → **实测只有 1 个任务（score）触达**：它是 scorer 的输入
#       （`backbone_features + scorer_fusion_proj(fused_feat_all)`，见 forward），
#       offset/embed 两个任务不经过它。
#       **决定：留在 SHARED 里不动。** 理由是量级 —— 它 1040 维 / SHARED 共 103281 维
#       ≈ 1%，对 cosine 的稀释 ≲0.5%；且 PCGrad 的修正项 `−(g_i·g_j/‖g_j‖²)·g_j` 作用到
#       `g_i` 的私有坐标时，`g_j` 在那里恰好是 0，不会跨任务注入（§③ 那个真问题）。
#       把它挪到 PRIVATE 只会省掉这 1% 的稀释，收益小于"与 plan 不一致"的记账成本。
#       每轮仍会打印实测触达数 + 一条自审计日志，若将来量级变大（比如该模块变宽）
#       就该重新考虑。
_ACGO_SHARED_PREFIXES = [
    "proj_gnn_to_fusion.", "fusion_mlp.", "backbone_proj.",
    "scorer_fusion_proj.", "growth_proj.", "causal_growth_attn.",
]

# ⚠️ 刻意**不**把 ACGO 诊断量放进 `loss_names`（曾被这么做，跑挂了）。
# `save_best_models_under_current_metrics`（model_checkpoint.py:295）对 tracking 里的
# **每一个** metric 名调 `find_func_from_metric_name`（子串匹配），找不到就抛
#     Exception: The metric name acgo_num_tasks doesn't have a func to measure ...
# 而补 tracker 的 `_metric_func` 又会给每个 metric 往 checkpoint 的 `models` 里加一个
# `best_*` 键。两条路互斥 ⇒ 诊断全部走 `[ACGO][epN]` 那条 logging.info 行（见
# `_acgo_backward` 末尾），可 grep、信息不丢。**别再把它塞回 loss_names。**


class PointGroup3heads_Hetero(BaseModel):
    """
    PointGroup with 3 heads + multi-view 2D branch + heterogeneous GNN fusion.

    Config options (in addition to PointGroup3heads):
        freeze_backbone: bool   — freeze the 3D Minkowski backbone (default True)
        hetero_rounds: int      — rounds of bidirectional message passing (default 2)
        superpoint_voxel: float — superpoint voxel size in metres (default 0.4)
        proj_resolution: int    — 2D projection image size (default 224)
    """

    __REQUIRED_DATA__ = ["pos"]
    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    # ──────────────────────────────────────────────
    #  __init__
    # ──────────────────────────────────────────────
    def __init__(self, option, model_type, dataset, modules):
        super().__init__(option)

        # ---- 3D Minkowski backbone (same as PointGroup3heads) ----
        backbone_options = option.get("backbone", {"architecture": "unet"})
        input_nc = dataset.feature_dimension
        if input_nc <= 0:
            # dataset.feature_dimension == 0 when processed cache has no data.x
            # Transforms add: XYZRelaFeature(3) + XYZFeature(add_z=True)(1) = 4
            input_nc = 4
        self.Backbone = Minkowski(
            backbone_options.get("architecture", "unet"),
            input_nc=input_nc,
            num_layers=4,
            config=backbone_options.get("config", {}),
        )

        # ---- Scorer (same as PointGroup3heads) ----
        self._scorer_type = option.get("scorer_type", None)
        cluster_voxel_size = False
        if cluster_voxel_size:
            self._voxelizer = GridSampling3D(
                cluster_voxel_size, quantize_coords=True, mode="mean", return_inverse=True
            )
        else:
            self._voxelizer = None
        self.ScorerUnet = Minkowski(
            "unet", input_nc=self.Backbone.output_nc, num_layers=4, config=option.scorer_unet
        )
        self.ScorerEncoder = Minkowski(
            "encoder", input_nc=self.Backbone.output_nc, num_layers=4, config=option.scorer_encoder
        )
        self.ScorerMLP = MLP([self.Backbone.output_nc, self.Backbone.output_nc, self.ScorerUnet.output_nc])
        self.ScorerHead = Seq().append(torch.nn.Linear(self.ScorerUnet.output_nc, 1)).append(torch.nn.Sigmoid())

        self.mask_supervise = option.get("mask_supervise", False)
        if self.mask_supervise:
            self.MaskScore = (
                Seq()
                .append(torch.nn.Linear(self.ScorerUnet.output_nc, self.ScorerUnet.output_nc))
                .append(torch.nn.ReLU())
                .append(torch.nn.Linear(self.ScorerUnet.output_nc, 1))
            )
        self.use_score_net = option.get("use_score_net", True)
        self.use_mask_filter_score_feature = option.get("use_mask_filter_score_feature", False)
        self.use_mask_filter_score_feature_start_epoch = option.get(
            "use_mask_filter_score_feature_start_epoch", 200
        )
        self.mask_filter_score_feature_thre = option.get("mask_filter_score_feature_thre", 0.5)
        self.cal_iou_based_on_mask = option.get("cal_iou_based_on_mask", False)
        self.cal_iou_based_on_mask_start_epoch = option.get("cal_iou_based_on_mask_start_epoch", 200)

        # ---- 3D heads ----
        # Semantic: reads pure backbone features (loaded from pretrained, trainable).
        # Offset/Embed: read fused features (backbone + GNN) for RGB-guided instances.
        fusion_out = 64
        self.Offset = Seq().append(MLP([fusion_out, fusion_out], bias=False))
        self.Offset.append(torch.nn.Linear(fusion_out, 3))
        self.Embed = Seq().append(MLP([fusion_out, fusion_out], bias=False))
        self.Embed.append(torch.nn.Linear(fusion_out, option.get("embed_dim", 5)))

        # Real growth feature (terrain-normalised canopy height, data.height).
        # Zero-init => exact no-op at start (== S1); learns to use the height
        # only if it helps the discriminative embedding / offset heads.
        self.use_growth_feature = option.get("use_growth_feature", False)
        # ★ 2026-10-24 新增：模块有效性实验里「生长先验」那半的对照开关。
        #   `dtm`（默认）= 数据集缓存的 h = z − DTM（treeins.py 用地面点建格网 DTM）；
        #   `raw_z`      = 直接用点自身的绝对高程 z，不做地形归一化。
        #   两者都走同一个 `growth_proj`（零初始化线性 + 残差），所以差别**只在输入**。
        #   ⚠️ 默认值保持 `dtm` ⇒ 老配置逐位不变。
        #   ★★ raw_z 取的是 batch 里的 `pos_z`（= XYZFeature 在 `Center` **之前**
        #      捕获的绝对高程，单位 cm，与 h 同量纲）；**不是** `raw_pos[:,2]` ——
        #      后者是逐样本减质心之后的高度（2026-10-25 修，见 forward 里的长注释）。
        self.growth_prior_type = str(option.get("growth_prior_type", "dtm")).lower()
        if self.growth_prior_type not in ("dtm", "raw_z"):
            raise ValueError(
                "[GrowthFeature] growth_prior_type 只能是 'dtm' 或 'raw_z'，"
                f"收到 {self.growth_prior_type!r}")
        if self.use_growth_feature and self.growth_prior_type == "raw_z":
            logging.getLogger(__name__).info(
                "[GrowthFeature] ★ 对照臂：使用**绝对高程 z**（batch.pos_z，Center 之前捕获），"
                "不做 DTM 地形归一化；起跑核对 source=raw_z_abs")
        # 只打一次的审计标记：height 缺失时整条生长分支会**静默**变成 no-op
        # （下面 forward 里 `if height is not None` 是安静跳过的），日志里看不出任何异常。
        # 而 height 来自数据集缓存（treeins.py 的 process() 写 data.height）—— 7 台机如果
        # 有一台的 processed_0.3 是旧版（早于 height 被写进 process()），机1/机5 就会
        # "看起来正常地"跑完却没有生长特征。这条日志是唯一的运行时证据。
        self._growth_feature_logged = False
        # ★ 2026-09-25 新增：逐轮诊断（见 backward 里的日志）。
        #   `_growth_res_probe` 在 forward 里赋值（不做 .item()，避免每步 device 同步）。
        self._growth_res_probe = None
        self._growth_proj_last_logged_epoch = -1
        if self.use_growth_feature:
            self.growth_proj = nn.Linear(1, fusion_out, bias=False)
            # ⚠️ 零初始化**不阻塞**学习：dL/dW = (dL/dres)ᵀ·h 与 W 无关 ⇒ W=0 时梯度
            #    照样非零。（因果模块当初不同：那里的门是**乘进**输出的，
            #    ∂out/∂A ∝ W_o = 0，两个零相乘才真把梯度掐断。）
            #    所以这里保留零初始化，只加诊断去**测**它有没有长起来。
            nn.init.zeros_(self.growth_proj.weight)

        self.Semantic = (
            Seq()
            .append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
            .append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes))
            .append(torch.nn.LogSoftmax(dim=-1))
        )

        self.loss_names = [
            "loss", "offset_norm_loss", "offset_dir_loss",
            "ins_loss", "ins_var_loss", "ins_dist_loss", "ins_reg_loss",
            "semantic_loss", "score_loss", "mask_loss", "merge_loss",
        ]
        stuff_classes = dataset.stuff_classes
        if is_list(stuff_classes):
            stuff_classes = torch.Tensor(stuff_classes).long()
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), stuff_classes])

        # ============================================================
        #  NEW: 2D branch + heterogeneous GNN fusion
        # ============================================================

        # Hyperparameters
        self.freeze_backbone = option.get("freeze_backbone", True)

        # ---- ★ A 开关（2³ 组合消融里 A=0 的那三格）----
        # `use_hetero=False` 把整个六视角异构分支旁路掉：
        #   - 不跑 multi_view_project / ResNet18；
        #   - GNN 压缩槽位固定喂 0 ⇒
        #       fused = fusion_mlp(cat(feat_b, 0)) + backbone_proj(feat_b)
        #   - **参数形状一律不变**：fusion_mlp 第一层仍是 [16+32, 128]。
        #     这一点是刻意的——A=0 与 A=1 的参数个数逐位相同，ACGO 的
        #     SHARED/PRIVATE 切分和冻结协议才对得上（见 acgo.py 文件头）。
        # ★ 默认 True ⇒ 所有既有 run（S1/S2/机3/机4）走的分支**逐位不变**。
        # ★ B 的因果那一半在 A=0 格子里自动不存在（它挂在 2D↔3D 边上，无 2D
        #   就没有那些边）⇒ A=0 的 B 退化为"只生长特征"，论文表里加脚注。
        #   这不是静默的：下面有互斥断言，配错会当场报错。
        self.use_hetero = option.get("use_hetero", True)
        if not self.use_hetero:
            logging.getLogger(__name__).info(
                "[Hetero] ★ use_hetero=False —— 六视角异构分支已旁路（A=0 格）："
                "不跑 2D 投影/ResNet18，GNN 槽位喂 0。")

        # ---- ★ A=0 的第二种实现：零化外观（2026-09-25 新增，用户选定）----
        # `use_hetero=False` 是"结构旁路"⇒ 因果注意力（挂在 2D↔3D 边上）**结构上
        # 不存在** ⇒ A=0 格子的 B 只能退化成"只生长特征"，B 列在表的两半不等价。
        #
        # 本开关是替代方案：**保留整条 2D 管线**（投影 / 建边 / 消息传递 / 因果
        # 注意力全部照跑），只把喂进 `multi_view_project` 的 rgb 清零。于是：
        #   - **因果注意力活着** ⇒ A=0 与 A=1 是同类模型，B 的效应在两半都可测；
        #   - A 因子变成"六视角**外观**信息的有无" ⇒ A 的对比是架构对齐的单变量对比。
        #
        # ⚠️ 必须如实写进论文：`multi_view_project` 生成的图像**只有 rgb 三个通道、
        #   没有 depth 通道**（深度只用于 `pixel_to_point` 决定每像素归哪个点）。
        #   所以零化 rgb 之后，**2D↔3D 的建边拓扑仍是真实几何**
        #   ⇒ 本开关的语义是"2D 分支在、但看不到外观"，**不是**"2D 分支不存在"。
        self.ablate_2d_appearance = option.get("ablate_2d_appearance", False)
        if self.ablate_2d_appearance and not self.use_hetero:
            raise ValueError(
                "[Hetero] ablate_2d_appearance=True 要求 use_hetero=True：本开关的"
                "实现是「保留 2D 管线、零化 rgb」，use_hetero=False 时 2D 管线整个"
                "不存在，本开关无意义。要 A=0 请二选一（结构旁路 use_hetero=False，"
                "或零化外观 use_hetero=True + ablate_2d_appearance=True），不要同时配。"
            )
        if self.ablate_2d_appearance:
            logging.getLogger(__name__).info(
                "[Hetero] ★ ablate_2d_appearance=True —— A=0 走「零化外观」："
                "2D 管线（投影/建边/GNN/因果注意力）全部保留，rgb 置 0 ⇒ "
                "看不到外观信息，但建边拓扑仍是真实几何。")

        self.hetero_rounds = option.get("hetero_rounds", 2)
        self.superpoint_voxel = option.get("superpoint_voxel", 0.4)
        self.proj_resolution = option.get("proj_resolution", 224)

        # ---- 2D 分支的尺度适配（微调泛化实验，2026-10-17）------------------
        # 两个常数都是按**源域样本尺度**定的：PUMPKIN 的样本是半径 8 的圆柱
        # （conf/data/panoptic/treeins_rad8_gs03.yaml:46 `radius: 8`），于是
        #     multiview radius = 8.0  （样本正好铺满一张 224×224 的图）
        #     superpoint_voxel = 0.4  （= 半外接尺寸的 5%，沿最长轴约 40 个超点）
        # 目标域的整块样本只有 0.4–6.3 m（ICRISAT 更小，且单位是毫米），沿用这两个
        # 常数会让 2D 分支几乎看不到东西、异构图几乎没有超点 —— 实测：ICRISAT 一个
        # 420 mm 的托盘投进 radius=8 的窗口会塌成几个像素，0.4 mm 的体素只出 1 个超点。
        #
        # scale_adapt=True 时按数据集自身的样本尺度等比缩放这两个量：
        #     ratio = sample_half_extent * pos_scale / 8.0
        # `sample_extent` 由数据集适配器提供（见 crops3d.py），是**缓存样本中位数**，
        # 不是逐样本量 —— 逐样本会让同一张图里不同 batch element 的语义词含义漂移。
        #
        # ★ 默认 False ⇒ 不碰任何已有配置，PUMPKIN 的 18 个 run 逐位不变。
        self.scale_adapt = bool(option.get("scale_adapt", False))
        self.multiview_radius = float(option.get("multiview_radius", 8.0))
        if self.scale_adapt:
            _train_ds = getattr(dataset, "train_dataset", None)
            _half_extent = getattr(_train_ds, "sample_extent", None)
            if _half_extent is None or float(_half_extent) <= 0:
                raise RuntimeError(
                    "[ScaleAdapt] dataset.train_dataset 没有可用的 `sample_extent`。"
                    "要么在适配器上实现该属性，要么把 scale_adapt 关掉。")
            _pos_scale = float(dataset.dataset_opt.get("pos_scale", 1.0))
            _ref = float(option.get("scale_adapt_reference_extent", 8.0))
            if _ref <= 0:
                raise ValueError("[ScaleAdapt] scale_adapt_reference_extent 必须 > 0")
            _ratio = (float(_half_extent) * _pos_scale) / _ref
            _base_spv = float(self.superpoint_voxel)
            _base_rad = self.multiview_radius
            self.superpoint_voxel = _base_spv * _ratio
            self.multiview_radius = _base_rad * _ratio
            logging.getLogger(__name__).info(
                "[ScaleAdapt] sample half-extent=%.6g (raw=%.6g, pos_scale=%.6g), "
                "reference=%.6g, ratio=%.6g -> superpoint_voxel %.6g->%.6g, "
                "multiview_radius %.6g->%.6g",
                float(_half_extent) * _pos_scale, float(_half_extent), _pos_scale,
                _ref, _ratio, _base_spv, self.superpoint_voxel, _base_rad, self.multiview_radius)

        # ---- 视角消融 (2026-10-05) ----
        # num_views < 6 保留**顶视优先**的前 N 个视角（顺序为何重要见
        # multiview.VIEW_ORDER_TOP_FIRST）；缺省或 >=6 = 全部六个 = 历史行为。
        # ★ 不写 num_views 的配置行为**逐位不变**——S0/S1/2³ 七格全部走原来的分支。
        self._num_views = option.get("num_views", NUM_VIEWS)
        # ★ 2026-10-24：视角顺序**显式化**。此前顺序只由 multiview.py 的默认值决定，
        #   而那个默认值 2026-10-08 从「顶视优先」改成了「顶视最后」⇒ 10-05/10-07
        #   跑的 view_v1 / view_v2 与新跑的 V3..V5 会是**两条不同的曲线**，拼起来无效。
        #   `view_order` 让每个 config 自己写死顺序并在日志里留证：
        #     top_last （默认，= 当前代码行为，逐位不变）：V1..V5 无顶视，只有 V6 有
        #     top_first（= 10-05..10-07 的旧顺序）：V1 就是最强的单视角（顶视）
        self._view_order = str(option.get("view_order", "top_last")).lower()
        if self._view_order == "top_first":
            self._view_dirs = select_views(self._num_views,
                                           order=VIEW_ORDER_TOP_FIRST)
        elif self._view_order == "top_last":
            self._view_dirs = select_views(self._num_views)
        else:
            raise ValueError(
                "[Hetero] view_order 只能是 'top_first' 或 'top_last'，"
                f"收到 {self._view_order!r}")
        if self._num_views != NUM_VIEWS:
            print("[Hetero] ★ view ablation: num_views=%d, order=%s, dirs=%s"
                  % (self._num_views, self._view_order, self._view_dirs))

        # Causal deconfounding + finetune scope
        self.use_causal = option.get("use_causal", False)
        self.use_causal_growth = option.get("use_causal_growth", False)
        self.unfreeze_scorer = option.get("unfreeze_scorer", False)
        self.unfreeze_msg = option.get("unfreeze_msg", False)

        # Causal instance merge (over-segment-then-merge). Consumes the
        # non-redundant height signal; trained with its own same-tree BCE loss.
        self.use_merge = option.get("use_merge", False)
        self.merge_module = CausalMerge() if self.use_merge else None
        self.merge_threshold = option.get("merge_threshold", 0.5)

        # Common message dimension
        D_msg = 128
        self.D_msg = D_msg  # stored so forward()/_compute_loss() can use it
        D_2d = 512  # ResNet18 output channels (layer4)

        # ---- 2D CNN: ResNet18, remove avgpool & fc ----
        from torchvision import models
        resnet = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet.children())[:-2])
        # Input:  [B, 3, 224, 224]
        # Output: [B, 512, 7, 7]

        # ---- Dimensionality projection ----
        self.proj_3d = nn.Linear(self.Backbone.output_nc, D_msg)
        self.proj_2d = nn.Linear(D_2d, D_msg)

        # ---- Heterogeneous message passing ----
        self.msg_3d_to_2d = nn.Sequential(
            nn.Linear(D_msg * 2, D_msg), nn.ReLU(), nn.Linear(D_msg, D_msg)
        )
        self.msg_2d_to_3d = nn.Sequential(
            nn.Linear(D_msg * 2, D_msg), nn.ReLU(), nn.Linear(D_msg, D_msg)
        )
        self.gru_3d = nn.GRUCell(D_msg, D_msg)
        self.gru_2d = nn.GRUCell(D_msg, D_msg)

        # ---- Causal deconfounding (optional) ----
        # Causal DAG: 3D geometry → 2D appearance (3D is the cause, 2D the
        # effect). A 2D patch feature therefore carries a part explainable by the
        # 3D superpoints it projects onto — confounded / redundant with LiDAR.
        # Before the 2D→3D message we regress that part out (backdoor
        # adjustment) so only the *novel* 2D signal (texture/colour) flows back.
        # Zero-init ⇒ no-op at start (= v5); it can only ever remove what is
        # predictable from 3D, so the novel 2D signal is kept by construction.
        if self.use_causal:
            # GLOBAL context confounder (Causal-SETR / Causal-SAM style), NOT a
            # per-sample MLP. A per-sample function of dst_3d is mathematically
            # redundant with the unfrozen msg_2d_to_3d MLP (composition of two
            # universal approximators on the same input = one), so it can only
            # reparameterise, never change expressivity. A low-rank global
            # dictionary instead PROJECTS OUT a shared "context" subspace from the
            # 2D feature, which msg_2d_to_3d cannot undo (that information is
            # gone) — a genuine information bottleneck, i.e. backdoor adjustment.
            self.causal_num_confounders = option.get("causal_num_confounders", 16)
            self.causal_loss_weight = option.get("causal_loss_weight", 0.1)
            self.confounder_dict = nn.Parameter(
                torch.randn(self.causal_num_confounders, D_msg) * 0.2
            )
            self.loss_names.extend(["causal_keep_ratio", "causal_decorr"])
            log = logging.getLogger(__name__)
            log.info("[Causal] Causal deconfounding enabled "
                     f"(global context dict, C={self.causal_num_confounders})")

        # ---- Causal Growth Attention (方案 A, optional) ----
        # Replaces the 2D→3D MLP message passing with a sparse causal-graph
        # attention (SPARTAN) + growth intervention token, plus a sparsity
        # regularisation that is the carrier of the causal structure. This is
        # the "constrain information flow" mechanism — NOT "estimate a signal
        # then modulate" (the root cause of the three prior failures).
        if self.use_causal_growth:
            # ★ 2026-10-24 新增：模块有效性实验的对照开关。
            #   `causal`（默认）= 本项目的稀疏因果门控 + 生长 token；
            #   `gat`   = Veličković et al. 2018 的标准多头图注意力（无门控、无 token）；
            #   `mlp`   = L0 均匀边平均（无注意力）：mlp_edge_attention.py
            #   `gcn`   = L1 对称归一化邻居聚合（含自环）：gcn_attention.py
            #   四者 forward 签名逐字相同 ⇒ 下游 `_causal_growth_cache`、稀疏度统计、
            #   ACGO 白名单都不用改。**默认值保持 causal ⇒ 老配置逐位不变。**
            #   ⚠️ 初始化口径：`mlp`/`gcn` 的 W_o 用 N(0,0.01)（与本方法一致），
            #      而 `gat` 臂沿用 `standard_attention.py` 的 xavier —— 该差异见
            #      各对照臂文件头，跨臂比较时必须单独交代。
            self.causal_growth_attn_type = str(
                option.get("causal_growth_attn_type", "causal")).lower()
            _ATTN_CHOICES = ("causal", "gat", "mlp", "gcn")
            if self.causal_growth_attn_type not in _ATTN_CHOICES:
                raise ValueError(
                    "[CausalGrowth] causal_growth_attn_type 只能是 "
                    f"{_ATTN_CHOICES} 之一，收到 {self.causal_growth_attn_type!r}")
            _heads = option.get("causal_growth_num_heads", 4)
            if self.causal_growth_attn_type == "gat":
                from .standard_attention import StandardGATAttention
                self.causal_growth_attn = StandardGATAttention(
                    D_msg=D_msg, num_heads=_heads,
                )
            elif self.causal_growth_attn_type == "mlp":
                from .mlp_edge_attention import MLPEdgeAttention
                self.causal_growth_attn = MLPEdgeAttention(
                    D_msg=D_msg, num_heads=_heads,
                )
            elif self.causal_growth_attn_type == "gcn":
                from .gcn_attention import GCNAttention
                self.causal_growth_attn = GCNAttention(
                    D_msg=D_msg, num_heads=_heads,
                )
            else:
                from .causal_growth_attention import CausalGrowthAttention
                self.causal_growth_attn = CausalGrowthAttention(
                    D_msg=D_msg,
                    num_heads=_heads,
                    tau=option.get("causal_growth_tau", 1.0),
                    use_hard_gumbel=option.get("causal_growth_hard_gumbel", False),
                    edge_prob_floor=option.get("causal_growth_edge_prob_floor", 0.05),
                )
            self.causal_sparse_weight = option.get("causal_sparse_weight", 0.01)
            self.loss_names.extend([
                "causal_sparse", "causal_edge_sparsity", "causal_token_sparsity",
            ])
            log = logging.getLogger(__name__)
            if self.causal_growth_attn_type == "gat":
                log.info("[CausalGrowth] ★ 对照臂：标准 GAT 注意力（无因果门控、无生长 token）"
                         f"(heads={_heads}, sparse_w={self.causal_sparse_weight})。"
                         "★ causal_edge_sparsity 在本臂 = 头平均 softmax 注意力权重，"
                         "**不是**门控概率，跨臂不可比。")
            elif self.causal_growth_attn_type in ("mlp", "gcn"):
                _desc = ("均匀边平均，无注意力" if self.causal_growth_attn_type == "mlp"
                         else "对称度归一化邻居聚合（含自环），无边权学习")
                log.info("[CausalGrowth] ★ 对照臂 L%s：%s"
                         "（W_o 用 N(0,0.01) 与本方法一致 ⇒ 起步近似 no-op）"
                         "★ causal_edge_sparsity 在本臂 = 常数边权/度数归一化系数，"
                         "**不是**学出来的注意力，跨臂不可比。"
                         % ("0" if self.causal_growth_attn_type == "mlp" else "1", _desc))
            else:
                log.info("[CausalGrowth] Causal growth attention enabled "
                         f"(heads={_heads}, "
                         f"sparse_w={self.causal_sparse_weight}, "
                         f"edge_prob_floor={option.get('causal_growth_edge_prob_floor', 0.05)}, "
                         f"masking=log-domain)")

        # ---- ACGO: 自适应梯度冲突引导协同优化（S3, optional）----
        # 不新增网络模块，只改**多任务梯度如何合成**：检测 pairwise 冲突 →
        # PCGrad 式投影（只在 SHARED 子空间内）→ 按近期冲突调制权重 → 写回
        # `p.grad`，交回原 Adam。实现细节与五处设计约束见 acgo.py 的文件头。
        #
        # `use_acgo=False`（默认）⇒ `backward` 走原路径 `self.loss.backward()`，
        # 与改动前逐位相同 —— 这是不许影响 S0/S1/S2 已有结果的前提。
        self.use_acgo = option.get("use_acgo", False)
        if self.use_acgo:
            # ★ 2026-10-24 新增：模块有效性实验里「第三个创新点」的对照开关。
            #   `acgo`（默认）  = 本项目的冲突投影 + EMA 自适应调制；
            #   `pcgrad`        = 只保留 PCGrad 投影（Yu et al. 2020），去掉 EMA 与自适应调制；
            #   `gradnorm`      = Chen et al. 2018 的梯度幅值归一化（**不做投影**）。
            #   ★ 三者中只有 `acgo` 是逐位不变的默认值；另两个都在 __init__ 里把
            #     投影/调制的组合**强制**成各自方法的定义，免得 config 写半套。
            self.acgo_variant = str(option.get("acgo_variant", "acgo")).lower()
            if self.acgo_variant not in ("acgo", "pcgrad", "gradnorm"):
                raise ValueError(
                    "[ACGO] acgo_variant 只能是 'acgo' / 'pcgrad' / 'gradnorm'，"
                    f"收到 {self.acgo_variant!r}")
            self.acgo_start_epoch = int(option.get("acgo_start_epoch", 32))
            self.acgo_projection = bool(option.get("acgo_projection", True))
            self.acgo_adaptive_weight = bool(option.get("acgo_adaptive_weight", True))
            if self.acgo_variant == "pcgrad":
                # PCGrad 的定义：有投影、无调制、无 EMA。
                self.acgo_projection = True
                self.acgo_adaptive_weight = False
            elif self.acgo_variant == "gradnorm":
                # GradNorm 的定义：没有投影（它是权重方法，不是几何方法）。
                self.acgo_projection = False
                self.acgo_adaptive_weight = False
                self.gradnorm_alpha = float(option.get("gradnorm_alpha", 0.15))
                self.gradnorm_lr = float(option.get("gradnorm_lr", 0.025))
                from .acgo import GradNormState
                self._gradnorm_state = GradNormState()
            self.acgo_ema_beta = float(option.get("acgo_ema_beta", 0.9))
            self.acgo_lambda = float(option.get("acgo_lambda", 1.0))
            self.acgo_eps = float(option.get("acgo_eps", 1e-8))
            # ★ 2026-09-26 新增：只在 `cos(g_i,g_j) < 阈值` 时才做 PCGrad 投影。
            #   默认 **0.0 = 原行为、逐位不变**（dot<0 ⟺ cos<0）。
            #   动机：实测 `mean_cos` 接近 0 ⇒ 现在的 `dot < 0` 会对**噪声级的负相关**
            #   也做一次投影，等于往合成梯度里注入随机扰动。设 −0.05 之类的容忍带
            #   只处理"真冲突"。★ 改了它就是**新的实验条件**，该行的旧数据不可混用。
            self.acgo_conflict_threshold = float(
                option.get("acgo_conflict_threshold", 0.0))
            self.acgo_shared_prefixes = [
                str(p) for p in option.get("acgo_shared_prefixes", _ACGO_SHARED_PREFIXES)
            ]
            # 任务集是**显式白名单**，不从 loss_names 推导（§④）。键 = 任务名，
            # 值 = 标称权重 w^0（**只**用于自适应权重的归一化记账，不乘梯度）。
            self._acgo_task_names = list(_ACGO_TASKS)
            # ★ 诊断量**不进 `loss_names`**（2026-09-24 实测踩坑后改）：
            #   `save_best_models_under_current_metrics` 对 tracking 里的**每一个** metric
            #   名调 `find_func_from_metric_name`（子串匹配），找不到就抛
            #   `Exception: The metric name acgo_num_tasks doesn't have a func ...`
            #   ⇒ 只进 loss_names 不补 tracker 的 `_metric_func`，**val 阶段第一个 epoch
            #   就会崩**（实测：S3-B 跑到 ep1 收尾时死）。
            #   而补 `_metric_func` 又会让每个 metric 往 checkpoint 的 `models` 里塞
            #   一个 `best_*` 键 —— plan 明确警告别动那张表。
            #   两条路互斥 ⇒ 正解是**根本不进 loss_names**：诊断全部走 `[ACGO][epN]`
            #   那条 logging.info 日志行（可 grep，信息不丢；见 _acgo_backward 末尾）。
            from .acgo import AcgoState
            self._acgo_state = AcgoState()
            self._acgo_tasks = OrderedDict()
            self._acgo_nominal = OrderedDict()
            self._acgo_last_logged_epoch = -1
            log = logging.getLogger(__name__)
            log.info(
                "[ACGO] enabled: variant=%s start_epoch=%d projection=%s adaptive=%s "
                "lambda=%.3f beta=%.2f conflict_threshold=%.4f shared_prefixes=%s"
                % (self.acgo_variant, self.acgo_start_epoch,
                   self.acgo_projection, self.acgo_adaptive_weight,
                   self.acgo_lambda, self.acgo_ema_beta,
                   self.acgo_conflict_threshold, self.acgo_shared_prefixes)
            )

        # ---- Fusion ----
        # GNN output compressed before fusion: D_msg→32, so
        # fusion_mlp sees cat(backbone[16], GNN[32]) instead of cat(16, 128).
        # Backbone contributes 16/48=33% of input energy instead of 16/144=11%.
        self.proj_gnn_to_fusion = nn.Linear(D_msg, 32)
        self.fusion_mlp = MLP([self.Backbone.output_nc + 32, 128, fusion_out])
        # Residual projection: backbone → fusion_out, zero-init so it
        # starts silent and gradually learns to pass backbone signal through.
        self.backbone_proj = nn.Linear(self.Backbone.output_nc, fusion_out)
        nn.init.zeros_(self.backbone_proj.weight)
        nn.init.zeros_(self.backbone_proj.bias)
        # Scorer input residual: fused(64) -> backbone-dim(16), zero-init so at
        # init the scorer still sees exactly the S0 backbone features (S0 scorer
        # weights stay meaningful); training then learns to inject the hetero
        # signal that separates the new false positives, which the frozen
        # backbone alone cannot distinguish.
        self.scorer_fusion_proj = nn.Linear(fusion_out, self.Backbone.output_nc)
        nn.init.zeros_(self.scorer_fusion_proj.weight)
        nn.init.zeros_(self.scorer_fusion_proj.bias)

        # ---- Cache for pixel→patch mapping (constant) ----
        self._patch_map = None  # built lazily on first forward

        # ============================================================
        #  NEW: Growth-Guided Causal Attention (optional)
        # ============================================================
        self.use_growth_attention = option.get("use_growth_attention", False)
        if self.use_growth_attention:
            from .growth_attention import (
                GrowthProgressEstimator,
                GrowthGuidedCrossAttention,
                GrowthConsistencyLoss,
            )

            H_max = option.get("growth_H_max", 120.0)
            k_growth = option.get("growth_k", 4.0)
            D_enc = option.get("growth_enc_dim", 16)
            D_growth = D_enc + 1  # sinusoidal enc + growth stage p

            self.growth_estimator = GrowthProgressEstimator(
                H_max=H_max,
                k_growth=k_growth,
                D_enc=D_enc,
                use_backbone=option.get("growth_use_backbone", False),
                D_backbone=self.Backbone.output_nc,
            )

            n_heads = option.get("growth_num_heads", 4)
            gamma = option.get("growth_gamma", 1.0)

            self.growth_attn_3d2d = GrowthGuidedCrossAttention(
                D_msg=D_msg, D_growth=D_growth,
                num_heads=n_heads, gamma=gamma,
            )
            self.growth_attn_2d3d = GrowthGuidedCrossAttention(
                D_msg=D_msg, D_growth=D_growth,
                num_heads=n_heads, gamma=gamma,
            )

            self.growth_loss_fn = GrowthConsistencyLoss(
                w_mono=option.get("growth_w_mono", 0.01),
                w_prior=option.get("growth_w_prior", 0.01),
                w_attn=option.get("growth_w_attn", 0.01),
            )
            self.growth_loss_weight = option.get("growth_loss_weight", 0.1)

            self.loss_names.extend([
                "growth_mono", "growth_prior",
                "growth_attn", "growth_total",
                "growth_attn_corr",
            ])

            log = logging.getLogger(__name__)
            log.info("[Growth] Growth-guided attention enabled "
                     f"(H_max={H_max}, k={k_growth}, heads={n_heads}, gamma={gamma})")

        # ---- ALFE: Allometric Feature Evolution (optional) ----
        self.use_allometric = option.get("use_allometric", False)
        if self.use_allometric:
            from .growth_attention import AllometricFeatureEvolution
            self.allometric = AllometricFeatureEvolution(
                feat_dim=fusion_out,
                K=option.get("allometric_K", 3),
                b0=tuple(option.get("allometric_b0", [1.3, 0.7, 1.0])),
                beta=option.get("allometric_beta", 1.0),
                gamma=option.get("allometric_gamma", 0.1),
            )
            self.loss_names.extend(["alfe_s", "alfe_s_std", "alfe_p_entropy"])
            log = logging.getLogger(__name__)
            log.info("[ALFE] Allometric feature evolution enabled "
                     f"(K={self.allometric.K}, b0={self.allometric.b0.tolist()}, "
                     f"beta={self.allometric.beta}, gamma={self.allometric.gamma})")

        # ---- ★ A=0 前置断言：这些模块与 use_hetero=False 互斥 ----
        # growth_attention / use_causal / causal_growth_attn 全部挂在 2D↔3D 异构边上
        # （见 forward 里的 `if edges_2d3d.shape[0] > 0:` 分支），A=0 时那些边不存在。
        # 静默跳过 = 配置写着"开"而实际没跑 —— 本项目已经吃过三次这种亏
        # （AddFeatsByKeys / BN eval 模式 / _freeze_for_ablation），所以直接报错。
        if not self.use_hetero:
            _mutex = [
                name for name, on in (
                    ("use_growth_attention", self.use_growth_attention),
                    ("use_causal", self.use_causal),
                    ("use_causal_growth", self.use_causal_growth),
                ) if on
            ]
            if _mutex:
                raise ValueError(
                    f"[Hetero] use_hetero=False 与 {_mutex} 互斥：这些模块都挂在 "
                    "2D↔3D 异构边上，而 A=0 的格子里没有那些边。请把它们关掉 —— "
                    "A=0 格的 B 因子退化为「只生长特征」（论文表里加脚注）。"
                    "★ 若你要的是「A=0 但因果注意力仍要跑」，正确配法是 "
                    "use_hetero=True + ablate_2d_appearance=True（保留 2D 管线、"
                    "零化 rgb），不是本项目前这种 use_hetero=False。"
                )

        # ---- Load pretrained shared weights (backbone + scorer + hetero fusion) ----
        pretrained_path = option.get("pretrained_3d_checkpoint", None)
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self._load_pretrained_3d(pretrained_path)
        elif pretrained_path is not None:
            # A configured-but-missing checkpoint used to be only a WARNING here.
            # That is dangerous rather than lenient: S1/S2 fine-tune with
            # freeze_backbone=True, so a silently-from-scratch run freezes a RANDOM
            # backbone and every result from it is meaningless -- while the log looks
            # completely normal. Fail loudly instead; a typo in the path costs two
            # GPU-days otherwise.
            raise FileNotFoundError(
                "[Hetero] pretrained_3d_checkpoint is set but does not exist:\n"
                f"    {pretrained_path}\n"
                "    Refusing to continue -- this run would silently train from "
                "scratch. Fix the path (note hydra rewrites cwd, so a relative path "
                "will not resolve) or unset the key."
            )

        # ---- Freeze everything except the ablation module(s) (growth / ALFE) ----
        # Must run AFTER all modules exist and AFTER weights are loaded.
        if option.get("freeze_pretrained", True) and (self.use_growth_attention or self.use_allometric or self.use_causal or self.use_causal_growth or self.use_growth_feature or self.use_merge or self.unfreeze_scorer or self.unfreeze_msg):
            self._freeze_for_ablation()

        # Growth outputs cache (populated in forward, consumed in _compute_loss)
        self._growth_cache = {}
        # ALFE outputs cache (populated in forward, consumed in _compute_loss)
        self._alfe_cache = {}
        # Causal deconfounding diagnostic cache
        self._causal_cache = {}
        # Causal growth attention cache (edge/token probabilities for sparsity)
        self._causal_growth_cache = {}
        # Causal merge cache (pair features + GT labels, populated in forward,
        # consumed in _compute_loss)
        self._merge_cache = {}

    # ──────────────────────────────────────────────
    #  set_input
    # ──────────────────────────────────────────────
    def set_input(self, data, device):
        self.raw_pos = data.pos.to(device)
        self.input = data
        all_labels = {lab: data[lab].to(device) for lab in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels)
        # ★ 2026-11-18 diagnostics (env-gated, zero cost otherwise): the 52-plot
        # retrain died inside MinkowskiEngine's first conv with cudaErrorIllegalAddress.
        # That kernel usually means the sparse-tensor INPUT is malformed, so dump
        # per-batch/per-sample facts before the forward. The last printed line
        # before the crash identifies the offending batch.
        if os.environ.get("PG_DEBUG_BATCH") == "1":
            import torch as _t
            pos = data.pos
            x = getattr(data, "x", None)
            b = getattr(data, "batch", None)
            print("[PG_DEBUG] pos=%s dtype=%s finite=%s"
                  % (tuple(pos.shape), pos.dtype, bool(_t.isfinite(pos).all())), flush=True)
            if x is not None:
                print("[PG_DEBUG] x  =%s dtype=%s finite=%s"
                      % (tuple(x.shape), x.dtype, bool(_t.isfinite(x).all())), flush=True)
            for name in ("rgb", "height"):
                v = getattr(data, name, None)
                if v is not None:
                    print("[PG_DEBUG] %-6s=%s dtype=%s finite=%s"
                          % (name, tuple(v.shape), v.dtype, bool(_t.isfinite(v).all())), flush=True)
            if b is not None and b.numel():
                cnt = _t.bincount(b).tolist()
                print("[PG_DEBUG] samples=%d points/sample=%s" % (len(cnt), cnt), flush=True)

    def get_opt_mergeTh(self):
        if self.opt.block_merge_th:
            return self.opt.block_merge_th
        return 0.01

    # ──────────────────────────────────────────────
    #  Load pretrained 3D backbone + scorer
    # ──────────────────────────────────────────────
    def _load_pretrained_3d(self, checkpoint_path):
        """Copy shared weights (backbone + scorer + heads + hetero fusion) from a
        checkpoint.

        Works with both a pure-3D PointGroup checkpoint (only the backbone /
        scorer / semantic keys match) and a Hetero v5 checkpoint (all shared
        keys match). Growth-attention modules (growth_*) are never present in
        the source checkpoint, so they stay randomly initialised.
        """
        ckp = torch.load(checkpoint_path, map_location="cpu")
        # Prefer best_F1 (best val F1) over best_mIoU, then latest.
        # `pretrained_weight_name` overrides that heuristic. It exists because the
        # three competitor models are initialised through `BaseModel.set_pretrained_weights`,
        # which takes the key verbatim from the config; without an override Ours would
        # start from `best_F1` while they start from whatever the config says. Pick a
        # key explicitly on BOTH paths so the fine-tune matrix is initialised the same way.
        # Default None = the historical heuristic, i.e. every existing config is unchanged.
        if "models" in ckp:
            models_dict = ckp["models"]
            preferred_name = self.opt.get("pretrained_weight_name", None)
            if preferred_name is not None:
                if preferred_name not in models_dict:
                    raise KeyError(
                        "[Hetero] pretrained_weight_name=%r not in checkpoint (available: %r)"
                        % (preferred_name, list(models_dict.keys())))
                preferred = preferred_name
            else:
                preferred = None
                for key in models_dict:
                    if "f1" in key.lower():
                        preferred = key
                        break
                if preferred is None:
                    for key in models_dict:
                        if "miou" in key.lower() or "iou" in key.lower():
                            preferred = key
                            break
                if preferred is None and "latest" in models_dict:
                    preferred = "latest"
            state_dict = models_dict.get(preferred, list(models_dict.values())[0])
            import logging
            log = logging.getLogger(__name__)
            log.info(f"[Hetero] Selected checkpoint key: {preferred}")
        else:
            state_dict = ckp

        model_dict = self.state_dict()
        # Everything shared between the source model and this model, except the
        # growth-attention modules. The MLP message passing (msg_*/gru_*) is
        # included: in the growth ablation it stays frozen and provides the base
        # features that the growth attention refines on top of.
        #
        # ⚠️ `causal_growth_attn.` 必须在名单里：S2 当初从 S1 起步时不需要它
        # （源 checkpoint 里没有这个模块），但 S3 要从 **S2 的 checkpoint** 起步，
        # 而 S2 的 checkpoint 里带着**训练好的**因果注意力权重。名单漏了它 ⇒
        # 那些权重被静默丢弃、重新随机初始化 ⇒ S3 不是真的 warm start。
        # 起跑核对：日志 `[Hetero] Copied N shared params` 的 N 必须 **大于**
        # S2 从 S1 起步时的 N；相同就是没加载上，立刻停。
        allowed_prefixes = (
            "Backbone.", "ScorerUnet.", "ScorerEncoder.",
            "ScorerMLP.", "ScorerHead.", "MaskScore.",
            "Semantic.", "Offset.", "Embed.",
            "resnet18.", "proj_3d.", "proj_2d.",
            "msg_3d_to_2d.", "msg_2d_to_3d.", "gru_3d.", "gru_2d.",
            "proj_gnn_to_fusion.", "fusion_mlp.", "backbone_proj.", "scorer_fusion_proj.",
            "growth_proj.", "causal_growth_attn.",
        )
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith(allowed_prefixes) and k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered[k] = v

        if filtered:
            model_dict.update(filtered)
            self.load_state_dict(model_dict, strict=False)
            import logging
            log = logging.getLogger(__name__)
            log.info("[Hetero] Copied %d/%d shared params from checkpoint",
                     len(filtered), len(state_dict))
            # `strict=False` swallows everything, so verify that the copy actually
            # landed. Five tensors is enough to distinguish "loaded" from "silently
            # skipped", and the model is still on CPU here.
            after = self.state_dict()
            probe = list(filtered.items())[:5]
            n_ok = sum(1 for k, v in probe if torch.equal(after[k].detach().cpu(), v.detach().cpu()))
            log.info("[Hetero] spot-check %d/%d copied tensors bit-equal", n_ok, len(probe))
            if probe and n_ok == 0:
                raise RuntimeError("[Hetero] load_state_dict did not apply the pretrained weights")
        else:
            raise RuntimeError(
                "[Hetero] No matching shared params found in checkpoint %s. "
                "Refusing to continue: the run would silently start from random weights."
                % checkpoint_path)

    def _freeze_for_ablation(self):
        """Freeze the v5 base and leave trainable only the enabled ablation
        module(s) PLUS the fusion path they feed into.

        Trainable: any enabled ablation module (growth_* / allometric.* /
        confounder_dict), plus proj_gnn_to_fusion + fusion_mlp +
        backbone_proj + Offset/Embed heads. Frozen: backbone, resnet18,
        semantic; the scorer (Scorer*) and v5 MLP message passing (msg_*/gru_*)
        are additionally unfrozen when unfreeze_scorer / unfreeze_msg are set.

        The growth residual is zero-init (W_o = 0) and ALFE is near-no-op at
        init, so the run still starts at (essentially) the v5 result. Unfreezing
        the fusion + clustering heads gives the ablation signal a trainable path
        to *improve* on v5 — in growth v2 they were frozen, so the residual could
        only perturb a frozen head (hurting F1) rather than be absorbed and
        exploited. The heavy feature extractors stay frozen to bound the downside.
        """
        if self.use_merge:
            # Merge-only finetune: the merge module is the ONLY trainable part.
            # Backbone, fusion tail, offset/embed, scorer, and the already-
            # finetuned growth_proj all stay frozen, so the merge can be ablated
            # cleanly against the growth-feature run (control = use_merge False).
            for name, param in self.named_parameters():
                param.requires_grad = name.startswith("merge_module.")
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            import logging
            log = logging.getLogger(__name__)
            log.info(f"[Freeze] Frozen {n_total - n_train} params; "
                     f"trainable {n_train} params (merge module only)")
            return
        trainable_prefixes = [
            "proj_gnn_to_fusion.", "fusion_mlp.", "backbone_proj.", "scorer_fusion_proj.",
            "Offset.", "Embed.",
        ]
        if self.use_growth_feature:
            trainable_prefixes += ["growth_proj."]
        if self.use_growth_attention:
            trainable_prefixes += [
                "growth_estimator.", "growth_attn_3d2d.", "growth_attn_2d3d.",
            ]
        if self.use_allometric:
            trainable_prefixes += ["allometric."]
        if self.use_causal:
            trainable_prefixes += ["confounder_dict"]
        if self.use_causal_growth:
            trainable_prefixes += ["causal_growth_attn."]
        if self.unfreeze_scorer:
            trainable_prefixes += [
                "ScorerUnet.", "ScorerEncoder.", "ScorerMLP.", "ScorerHead.",
            ]
        if self.unfreeze_msg:
            trainable_prefixes += [
                "msg_3d_to_2d.", "msg_2d_to_3d.", "gru_3d.", "gru_2d.",
                "proj_3d.", "proj_2d.",
            ]
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith(tuple(trainable_prefixes))
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        import logging
        log = logging.getLogger(__name__)
        log.info(f"[Freeze] Frozen {n_total - n_train} params; "
                 f"trainable {n_train} params (ablation + fusion + offset/embed)")

    # ──────────────────────────────────────────────
    #  forward
    # ──────────────────────────────────────────────
    # noinspection PyTypeChecker
    def forward(self, epoch=-1, **kwargs):
        device = self.raw_pos.device
        # ★ 2026-11-19: per-batch NaN flag. Set by the clustering guards below when
        # a batch produced non-finite embeddings/offsets; consumed by the scorer
        # call (skip) and by `backward` (skip the optimizer step). Reset here so a
        # stale flag can never leak into the next batch.
        self._bad_batch = False

        # ------------------------------------------------------------
        # 1. Multi-view projection  (per-batch-element)   [A=0：跳过]
        # ------------------------------------------------------------
        if self.use_hetero:
            mv_images, pixel_to_point_all = self._run_multiview_projection(device)
        else:
            mv_images = pixel_to_point_all = None

        # ------------------------------------------------------------
        # 2. 3D Backbone
        # ------------------------------------------------------------
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_features = self.Backbone(self.input).x
        else:
            backbone_features = self.Backbone(self.input).x
        # backbone_features: [N, D_backbone]

        # ------------------------------------------------------------
        # 3. 2D ResNet18  (batch all views from all batch elements)   [A=0：跳过]
        # ------------------------------------------------------------
        if self.use_hetero:
            # mv_images: [B, 6, 3, R, R]  →  flatten to [B*6, 3, R, R]
            B = mv_images.shape[0]
            flat_images = mv_images.reshape(
                B * self._num_views, 3, self.proj_resolution, self.proj_resolution)
            img_feats = self.resnet18(flat_images)                     # [B*6, 512, 7, 7]
            img_feats = img_feats.reshape(B, self._num_views, 512, 7, 7)     # [B, 6, 512, 7, 7]
        else:
            img_feats = None

        # ------------------------------------------------------------
        # 4. Process each batch element independently for graph parts
        # ------------------------------------------------------------
        batch_ids = getattr(self.input, 'batch', torch.zeros(self.raw_pos.shape[0], dtype=torch.long, device=device))
        unique_b = torch.unique(batch_ids)

        fused_feat_list = []

        for b_idx in unique_b:
            mask_b = batch_ids == b_idx
            idx_b = torch.where(mask_b)[0]

            pos_b = self.raw_pos[mask_b]            # [n_b, 3]
            feat_b = backbone_features[mask_b]      # [n_b, D_backbone]

            # ---- ★ A=0（2³ 消融里 A 关闭的三格）：旁路整个 2D / 异构分支 ----
            # 不走 superpoint / 多视图投影 / 建边 / 消息传递，GNN 压缩槽位直接喂 0：
            #   fused = fusion_mlp(cat(feat_b, 0)) + backbone_proj(feat_b)
            # 形状与 A=1 格逐位一致（fusion_mlp 第一层仍是 [16+32, 128]），
            # 所以参数个数、ACGO 的 SHARED/PRIVATE 切分都不变。
            if not self.use_hetero:
                gnn_slot = torch.zeros(
                    feat_b.shape[0], self.proj_gnn_to_fusion.out_features,
                    device=device, dtype=feat_b.dtype,
                )
                fused_b = self.fusion_mlp(torch.cat([feat_b, gnn_slot], dim=-1)) \
                          + self.backbone_proj(feat_b)
                if self.use_allometric:
                    fused_b, alfe_p, alfe_s = self.allometric(fused_b)
                    self._alfe_cache[b_idx.item()] = {"s": alfe_s, "p": alfe_p}
                fused_feat_list.append(fused_b)
                continue
            rgb_b = self.input.rgb[mask_b]          # [n_b, 3]

            # ---- 4a. Superpoint voxelisation ----
            sp_ids = self._generate_superpoints(pos_b)         # [n_b]
            M_b = int(sp_ids.max().item()) + 1

            sp_feat_3d = scatter(feat_b, sp_ids, dim=0, reduce='mean')  # [M_b, D_backbone]

            # ---- 4b. Project to common dim ----
            sp_feat_3d_proj = self.proj_3d(sp_feat_3d)                 # [M_b, D_msg]

            patch_feats_2d = img_feats[b_idx]                          # [6, 512, 7, 7]
            patch_feats_2d = patch_feats_2d.permute(0, 2, 3, 1)       # [6, 7, 7, 512]
            patch_feats_2d = patch_feats_2d.reshape(self._num_views * 49, 512)  # [294, 512]
            patch_feats_2d_proj = self.proj_2d(patch_feats_2d)        # [294, D_msg]

            # ---- 4c. Build heterogeneous edges ----
            pixel_to_point = pixel_to_point_all[b_idx]                 # [6, R²]
            edges_3d2d, edges_2d3d = self._build_hetero_edges(
                pixel_to_point, sp_ids
            )

            # ---- 4d. Heterogeneous message passing ----
            # Stage 1: v5 MLP message passing (frozen in the growth ablation),
            # providing the base 3D↔2D features.
            for _ in range(self.hetero_rounds):
                # 3D → 2D
                if edges_3d2d.shape[0] > 0:
                    src_3d = sp_feat_3d_proj[edges_3d2d[:, 0]]
                    dst_2d = patch_feats_2d_proj[edges_3d2d[:, 1]]
                    msg_3d2d = self.msg_3d_to_2d(torch.cat([src_3d, dst_2d], dim=-1))
                    agg_2d = scatter(msg_3d2d, edges_3d2d[:, 1], dim=0,
                                     reduce='mean', dim_size=self._num_views * 49)
                    patch_feats_2d_proj = self.gru_2d(agg_2d, patch_feats_2d_proj)

                # 2D → 3D
                if edges_2d3d.shape[0] > 0:
                    src_2d = patch_feats_2d_proj[edges_2d3d[:, 0]]
                    dst_3d = sp_feat_3d_proj[edges_2d3d[:, 1]]

                    if self.use_causal_growth:
                        # 方案 A（修正）：因果生长注意力作为「残差」加在 v5 的稠密
                        # MLP 聚合之上，而非替换。W_o/W_og 零初始化 → 起步
                        # agg_3d = dense + 0 = S1 的稠密消息（真 no-op），模型从
                        # S1 结果出发，因果注意力只能在此基础上细化。因果结构
                        # （可学习稀疏邻接）约束的是「残差信息流」走哪条边。
                        msg_2d3d = self.msg_2d_to_3d(torch.cat([src_2d, dst_3d], dim=-1))
                        agg_dense = scatter(msg_2d3d, edges_2d3d[:, 1], dim=0,
                                            reduce='mean', dim_size=M_b)
                        agg_res, edge_prob, token_prob = self.causal_growth_attn(
                            src_2d, dst_3d, sp_feat_3d_proj, edges_2d3d, M_b,
                        )
                        self._causal_growth_cache[b_idx.item()] = {
                            "edge_prob": edge_prob,
                            "token_prob": token_prob,
                        }
                        agg_3d = agg_dense + agg_res
                        sp_feat_3d_proj = self.gru_3d(agg_3d, sp_feat_3d_proj)

                    else:
                        if self.use_causal:
                            # Backdoor adjustment: project the 2D patch feature OFF the
                            # global context subspace (confounder dict), keeping only the
                            # novel 2D signal (texture/colour) for the 2D→3D message.
                            sim = src_2d @ self.confounder_dict.T / (self.D_msg ** 0.5)
                            w = torch.softmax(sim, dim=-1)                 # [E, C]
                            ctx_comp = w @ self.confounder_dict           # [E, D_msg]
                            src_2d_causal = src_2d - ctx_comp
                            keep = src_2d_causal.norm(dim=-1) / (src_2d.norm(dim=-1) + 1e-8)
                            self._causal_cache[b_idx.item()] = {
                                "keep": keep.mean(),
                                "src2d": src_2d,
                                "dst3d": dst_3d,
                            }
                            src_2d = src_2d_causal
                        msg_2d3d = self.msg_2d_to_3d(torch.cat([src_2d, dst_3d], dim=-1))
                        agg_3d = scatter(msg_2d3d, edges_2d3d[:, 1], dim=0,
                                         reduce='mean', dim_size=M_b)
                        sp_feat_3d_proj = self.gru_3d(agg_3d, sp_feat_3d_proj)

            # Stage 2 (optional): Growth-guided attention as a zero-initialised
            # residual refinement on top of the v5 message passing. W_o is
            # zero-init, so at init this is a no-op — the model starts at the
            # v5 result and the attention can only refine it.
            if self.use_growth_attention:
                # --- Growth progress estimation ---
                # Superpoint positions + per-superpoint point count (real
                # density signal, not the constant placeholder that starved the
                # estimator and made growth_mono collapse to ~0).
                sp_pos = scatter(pos_b, sp_ids, dim=0, reduce='mean')   # [M_b, 3]
                sp_count = scatter(
                    torch.ones(pos_b.shape[0], device=device),
                    sp_ids, dim=0, reduce='sum',
                )                                                        # [M_b]
                g_sp, sp_p, sp_r = self.growth_estimator(
                    sp_pos, sp_feat_3d, sp_count=sp_count,
                )
                # g_sp: [M_b, D_enc+1],  sp_p: [M_b, 1],  sp_r: [M_b, 1]

                # For 2D patches: estimate growth from connected superpoints
                patch_p = torch.zeros(self._num_views * 49, 1, device=device)
                if edges_3d2d.shape[0] > 0:
                    patch_p = scatter(
                        sp_p[edges_3d2d[:, 0]], edges_3d2d[:, 1],
                        dim=0, reduce='mean', dim_size=self._num_views * 49,
                    )
                    patch_p = patch_p.clamp(0.0, 1.0)

                # Patch growth encoding (sinusoidal + raw stage p)
                g_patch = torch.cat([
                    self.growth_estimator._sinusoidal_encode(patch_p),
                    patch_p,
                ], dim=-1)  # [294, D_enc+1]

                # --- Growth-guided residual attention ---
                attns_3d2d_list = []
                attns_2d3d_list = []
                corrs_3d2d_list = []
                corrs_2d3d_list = []

                for _ in range(self.hetero_rounds):
                    # 3D → 2D: growth-guided cross-attention (residual)
                    if edges_3d2d.shape[0] > 0:
                        patch_feats_2d_proj, a_3d2d, corr_3d2d = self.growth_attn_3d2d(
                            src_feat=sp_feat_3d_proj,
                            dst_feat=patch_feats_2d_proj,
                            growth_src=g_sp,
                            growth_dst=g_patch,
                            edges=edges_3d2d,
                            M_dst=self._num_views * 49,
                        )
                        attns_3d2d_list.append(a_3d2d)
                        corrs_3d2d_list.append(corr_3d2d)

                    # 2D → 3D: growth-guided cross-attention (residual)
                    if edges_2d3d.shape[0] > 0:
                        sp_feat_3d_proj, a_2d3d, corr_2d3d = self.growth_attn_2d3d(
                            src_feat=patch_feats_2d_proj,
                            dst_feat=sp_feat_3d_proj,
                            growth_src=g_patch,
                            growth_dst=g_sp,
                            edges=edges_2d3d,
                            M_dst=M_b,
                        )
                        attns_2d3d_list.append(a_2d3d)
                        corrs_2d3d_list.append(corr_2d3d)

                # Cache growth outputs for loss
                self._growth_cache[b_idx.item()] = {
                    "p": sp_p,
                    "r": sp_r,
                    "attn_3d2d": attns_3d2d_list[-1] if attns_3d2d_list else None,
                    "attn_2d3d": attns_2d3d_list[-1] if attns_2d3d_list else None,
                    "corr_3d2d": corrs_3d2d_list[-1] if corrs_3d2d_list else None,
                    "corr_2d3d": corrs_2d3d_list[-1] if corrs_2d3d_list else None,
                    "sp_ids": sp_ids,
                }

            # ---- 4e. Upsample superpoint → point level + compress for fusion ----
            point_sp_feat = sp_feat_3d_proj[sp_ids]                    # [n_b, D_msg]
            point_sp_feat_compact = self.proj_gnn_to_fusion(point_sp_feat)  # [n_b, 32]

            # ---- 4f. Fuse: backbone[16] + compressed GNN[32] = 48 dims ----
            fused_b = self.fusion_mlp(torch.cat([feat_b, point_sp_feat_compact], dim=-1)) \
                      + self.backbone_proj(feat_b)
            # fused_b: [n_b, fusion_out]

            # ---- 4g. Allometric Feature Evolution (ALFE, optional) ----
            # Per-sample global allometric scaling of the fused features before
            # the Offset/Embed heads (MVCTNet §3.2). One scalar per sample,
            # learned from that sample's own organ composition.
            if self.use_allometric:
                fused_b, alfe_p, alfe_s = self.allometric(fused_b)
                self._alfe_cache[b_idx.item()] = {"s": alfe_s, "p": alfe_p}

            fused_feat_list.append(fused_b)

        # ------------------------------------------------------------
        # 5. Concatenate all batch elements
        # ------------------------------------------------------------
        fused_feat_all = torch.cat(fused_feat_list, dim=0)           # [N, fusion_out]

        # Real growth feature injection: add a terrain-normalised canopy-height
        # residual to the fused features before the Offset/Embed (and scorer)
        # heads. growth_proj is zero-init => exact no-op at start (= S1).
        if self.use_growth_feature:
            # ★ 2026-10-24：对照臂用绝对高程，其余逐位不变。
            # ★★ 2026-10-25 修正（真 bug）：原来这里取 `self.raw_pos[:, 2:3]`，
            #   但 `raw_pos = data.pos` 是**经过 `Center` 之后**的坐标
            #   （PyG 的 Center：逐样本减去本样本质心，见数据 config 的
            #   train_transforms 顺序），所以它给的是「样本内居中高度」，
            #   **不是绝对高程**。而 config 头、日志、checkpoint 里全写着"绝对高程 z"。
            #   链路里唯一能拿到绝对 z 的是 `data.pos_z`：由
            #   `XYZFeature(add_z: True)` 在 `Center` **之前**捕获
            #   （shuffle / 体素下采样都会同步处理它，见 grid_transform.shuffle_data）。
            #   ⇒ raw_z 臂改用 pos_z；**拿不到就报错**，绝不静默退回 raw_pos
            #     （那样会把「绝对高程 vs DTM 归一化」悄悄变成「居中高度 vs DTM 归一化」，
            #      两者结论完全不同）。
            height, _src = resolve_growth_prior(
                self.input, self.raw_pos, self.growth_prior_type)
            _h_ok = height is not None and height.shape[0] == fused_feat_all.shape[0]
            if not self._growth_feature_logged:
                # ★ 这一行是"生长特征真的生效了"的唯一运行时证据（见 __init__ 里的注释）。
                # 起跑核对：机1/机5 必须出现 `height PRESENT`；出现 MISSING 立刻停 ——
                # 那说明该机的 processed_0.3 是旧版缓存，生长分支全程是 no-op。
                # raw_z 臂还要核对 `source=raw_z_abs` —— 出现别的 source 说明取错了先验。
                self._growth_feature_logged = True
                logging.getLogger(__name__).info(
                    "[GrowthFeature] height %s (n=%s vs fused=%s, source=%s) "
                    "—— use_growth_feature=True"
                    % ("PRESENT" if _h_ok else "★MISSING★",
                       None if height is None else int(height.shape[0]),
                       int(fused_feat_all.shape[0]),
                       _src))
            if _h_ok:
                _gres = self.growth_proj(height.to(device))
                # 残差幅度的探针：只做一次 reduce、**不** .item()（避免每步 device
                # 同步），backward 里每轮读一次。这是"生长模块是活的还是死的"的
                # 唯一直接证据 —— 若 |残差| 长期 ≈0，整条生长分支等于 no-op。
                self._growth_res_probe = _gres.detach().abs().mean()
                fused_feat_all = fused_feat_all + _gres


        # ---- Heads ----
        # Semantic: pure backbone features (loaded from pretrained, trainable)
        semantic_logits = self.Semantic(backbone_features)          # [N, num_classes]
        # Offset / Embed: fused features (backbone + GNN, for RGB-guided instances)
        offset_logits = self.Offset(fused_feat_all)                 # [N, 3]
        embed_logits = self.Embed(fused_feat_all)                   # [N, embed_dim]

        # ------------------------------------------------------------
        # 6. Clustering + scoring  (same logic as PointGroup3heads)
        # ------------------------------------------------------------
        cluster_scores = None
        mask_scores = None
        all_clusters = None
        cluster_type = None

        if self.use_score_net:
            if epoch > self.opt.prepare_epoch:
                cluster_fn = getattr(self, f'_cluster{self.opt.cluster_type}', None)
                if cluster_fn is not None:
                    all_clusters, cluster_type = cluster_fn(semantic_logits, offset_logits, embed_logits)
                if self.use_merge and all_clusters and len(all_clusters):
                    all_clusters, cluster_type = self._merge_step(all_clusters, cluster_type)
                if all_clusters and len(all_clusters) and not self._bad_batch:
                    cluster_scores, mask_scores = self._compute_score(
                        epoch, all_clusters,
                        backbone_features + self.scorer_fusion_proj(fused_feat_all),
                        semantic_logits,
                    )
                elif self._bad_batch:
                    # ★ 2026-11-19: the scorer is a Minkowski UNet; feeding it the
                    # degenerate clusters produced from a sanitised NaN batch is
                    # what turned the original ValueError into a CUDA illegal
                    # address. Skip scoring for this batch entirely.
                    logging.getLogger(__name__).warning(
                        "[NaNGuard] bad batch -> scorer skipped (clusters=%s)",
                        None if all_clusters is None else len(all_clusters))
        else:
            with torch.no_grad():
                if epoch % 1 == 0:
                    cluster_fn = getattr(self, f'_cluster{self.opt.cluster_type}', None)
                    if cluster_fn is not None:
                        all_clusters, cluster_type = cluster_fn(semantic_logits, offset_logits, embed_logits)

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            embed_logits=embed_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            mask_scores=mask_scores,
            cluster_type=cluster_type,
        )

    # ──────────────────────────────────────────────
    #  Multi-view projection helper
    # ──────────────────────────────────────────────
    def _run_multiview_projection(self, device):
        """Run multi-view projection per batch element. Returns images + pixel maps."""
        batch_ids = getattr(
            self.input, 'batch',
            torch.zeros(self.raw_pos.shape[0], dtype=torch.long, device=device)
        )
        unique_b = torch.unique(batch_ids)

        all_images = []
        all_maps = []

        for b in unique_b:
            mask = batch_ids == b
            pos_b = self.raw_pos[mask]
            rgb_b = self.input.rgb[mask].to(device)
            if self.ablate_2d_appearance:
                # ★ A=0（零化外观）：只把颜色信息抹掉，**不动** pos ⇒ 下面
                # multi_view_project 里的 pixel_to_point（每像素归哪个点）用的
                # 仍是真实几何，建边拓扑不变。详见 __init__ 里本开关的注释。
                rgb_b = torch.zeros_like(rgb_b)
            imgs, p2p = multi_view_project(pos_b, rgb_b,
                                           resolution=self.proj_resolution,
                                           radius=self.multiview_radius,
                                           view_dirs=self._view_dirs)
            all_images.append(imgs)
            all_maps.append(p2p)

        return torch.stack(all_images, dim=0), torch.stack(all_maps, dim=0)

    # ──────────────────────────────────────────────
    #  Superpoint generation
    # ──────────────────────────────────────────────
    def _generate_superpoints(self, pos):
        """Assign each point to a superpoint via coarse voxelisation."""
        voxel_coords = (pos / self.superpoint_voxel).floor().long()  # [N, 3]
        _, sp_ids = torch.unique(voxel_coords, dim=0, return_inverse=True)
        return sp_ids  # [N]

    # ──────────────────────────────────────────────
    #  Heterogeneous edge construction
    # ──────────────────────────────────────────────
    def _build_hetero_edges(self, pixel_to_point, superpoint_ids):
        """
        Build 3D-superpoint ↔ 2D-patch edges from projection mapping.

        Args:
            pixel_to_point: [6, R²]  point index per (view, flat_pixel), -1 = empty
            superpoint_ids:  [N]     superpoint assignment per point

        Returns:
            edges_3d2d: [E, 2]  (superpoint_idx, patch_idx)
            edges_2d3d: [E, 2]  (patch_idx, superpoint_idx)
        """
        device = pixel_to_point.device

        # Lazily build patch map (constant across forward calls)
        if self._patch_map is None or self._patch_map.device != device:
            self._patch_map = pixel_to_patch_map(self.proj_resolution).to(device)
        patch_map = self._patch_map  # [R²]

        valid = pixel_to_point >= 0                     # [6, R²]
        views, pixels = torch.where(valid)              # [E_raw]
        if views.numel() == 0:
            empty = torch.empty((0, 2), dtype=torch.long, device=device)
            return empty, empty

        point_idx = pixel_to_point[valid]               # [E_raw]
        sp_idx = superpoint_ids[point_idx]              # [E_raw]
        patch_idx = views * 49 + patch_map[pixels]      # [E_raw]

        edges = torch.stack([sp_idx, patch_idx], dim=1)  # [E_raw, 2]
        edges_3d2d = torch.unique(edges, dim=0)

        edges_2d3d = torch.stack([edges_3d2d[:, 1], edges_3d2d[:, 0]], dim=1)

        return edges_3d2d, edges_2d3d

    # ══════════════════════════════════════════════
    #  Clustering methods  (same as PointGroup3heads)
    # ══════════════════════════════════════════════

    def _cluster1(self, semantic_logits, offset_logits, embed_logits=None):
        """offset-based region_grow"""
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_votes = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10,
        )
        all_clusters = [c.to(self.device) for c in clusters_votes]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8, device=self.device)
        return all_clusters, cluster_type

    def _cluster2(self, semantic_logits, offset_logits, embed_logits=None):
        """pos-based + offset-based region_grow"""
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_pos = region_grow(
            self.raw_pos, predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            min_cluster_size=10,
        )
        clusters_votes = region_grow(
            self.raw_pos + offset_logits, predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10,
        )
        all_clusters = clusters_pos + clusters_votes
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8, device=self.device)
        if len(clusters_pos):
            cluster_type[len(clusters_pos):] = 1
        return all_clusters, cluster_type

    def _cluster3(self, semantic_logits, offset_logits, embed_logits):
        """embedding-based MeanShift"""
        predicted_labels = torch.max(semantic_logits, 1)[1]
        N = embed_logits.shape[0]
        ind = torch.arange(0, N)
        ignore_labels = self._stuff_classes.to(self.device)
        label_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        for lab in torch.unique(predicted_labels):
            if lab in ignore_labels:
                label_mask[predicted_labels == lab] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        embeds_u = embed_logits[label_mask]
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(
            embeds_u, unique_in_batch, label_batch, local_ind, 0, self.opt.bandwidth
        )
        all_clusters = [c.to(self.device) for c in clusters_embed]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8, device=self.device)
        return all_clusters, cluster_type

    def _cluster4(self, semantic_logits, offset_logits, embed_logits):
        """pos-based region_grow + embedding-based MeanShift"""
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_pos = region_grow(
            self.raw_pos, predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            min_cluster_size=10,
        )
        N = embed_logits.shape[0]
        ind = torch.arange(0, N)
        ignore_labels = self._stuff_classes.to(self.device)
        label_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        for lab in torch.unique(predicted_labels):
            if lab in ignore_labels:
                label_mask[predicted_labels == lab] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        embeds_u = embed_logits[label_mask]
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(
            embeds_u, unique_in_batch, label_batch, local_ind, 1, self.opt.bandwidth
        )
        all_clusters = clusters_pos + clusters_embed
        cluster_type = list(np.zeros(len(clusters_pos), dtype=np.uint8)) + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type, device=self.device)
        return all_clusters, cluster_type

    def _cluster5(self, semantic_logits, offset_logits, embed_logits):
        """offset-based region_grow + embedding-based MeanShift"""
        # ★ 2026-11-18 NaN guard: a rare training batch produced non-finite
        # embeddings, and sklearn's MeanShift rejects them (ValueError: Input
        # contains NaN, infinity...). Sanitising here keeps the forward alive; the
        # loss-side guard in base_model.optimize_parameters2 then skips the update
        # for that batch instead of letting NaN gradients reach the weights.
        if not torch.isfinite(embed_logits).all():
            logging.getLogger(__name__).warning("[NaNGuard] non-finite embed_logits in _cluster5 -> sanitized")
            embed_logits = torch.nan_to_num(embed_logits, nan=0.0, posinf=0.0, neginf=0.0)
            self._bad_batch = True
        if not torch.isfinite(offset_logits).all():
            logging.getLogger(__name__).warning("[NaNGuard] non-finite offset_logits in _cluster5 -> sanitized")
            offset_logits = torch.nan_to_num(offset_logits, nan=0.0, posinf=0.0, neginf=0.0)
            self._bad_batch = True
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_pos = region_grow(
            self.raw_pos + offset_logits, predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10,
        )
        N = embed_logits.shape[0]
        ind = torch.arange(0, N)
        ignore_labels = self._stuff_classes.to(self.device)
        label_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        for lab in torch.unique(predicted_labels):
            if lab in ignore_labels:
                label_mask[predicted_labels == lab] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        embeds_u = embed_logits[label_mask]
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(
            embeds_u, unique_in_batch, label_batch, local_ind, 1, self.opt.bandwidth
        )
        all_clusters = clusters_pos + clusters_embed
        cluster_type = list(np.zeros(len(clusters_pos), dtype=np.uint8)) + cluster_type_embeds
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type, device=self.device)
        return all_clusters, cluster_type

    def _cluster6(self, semantic_logits, offset_logits, embed_logits):
        """pos-based + offset-based region_grow + embedding-based MeanShift"""
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_pos = region_grow(
            self.raw_pos, predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            min_cluster_size=10,
        )
        clusters_votes = region_grow(
            self.raw_pos + offset_logits, predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=10,
        )
        N = embed_logits.shape[0]
        ind = torch.arange(0, N)
        ignore_labels = self._stuff_classes.to(self.device)
        label_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        for lab in torch.unique(predicted_labels):
            if lab in ignore_labels:
                label_mask[predicted_labels == lab] = False
        local_ind = ind[label_mask]
        label_batch = self.input.batch[label_mask]
        unique_in_batch = torch.unique(label_batch)
        embeds_u = embed_logits[label_mask]
        clusters_embed, cluster_type_embeds = meanshift_cluster.cluster_single(
            embeds_u, unique_in_batch, label_batch, local_ind, 2, self.opt.bandwidth
        )
        all_clusters = clusters_pos + clusters_votes + clusters_embed
        cluster_type = (
            list(np.zeros(len(clusters_pos), dtype=np.uint8))
            + list(np.ones(len(clusters_votes), dtype=np.uint8))
            + cluster_type_embeds
        )
        all_clusters = [c.clone().detach().to(self.device) for c in all_clusters]
        cluster_type = torch.tensor(cluster_type, device=self.device)
        return all_clusters, cluster_type

    # ══════════════════════════════════════════════
    #  Causal instance merge  (over-segment-then-merge)
    # ══════════════════════════════════════════════

    def _merge_step(self, all_clusters, cluster_type):
        """Post-clustering causal merge.

        Training: compute same-tree pair features + GT labels for the merge loss
        (stored in self._merge_cache); clusters are left unmerged so the frozen
        scorer's score loss stays consistent with its own training distribution.
        Eval: apply the learned merge via union-find and return merged clusters.
        """
        device = self.raw_pos.device
        height = getattr(self.input, "height", None)
        if height is None:
            return all_clusters, cluster_type
        height = height.to(device).view(-1)
        batch = self.input.batch.to(device)
        centroids, h_mean, h_max, h_min, xy_min, xy_max, sizes, cbatch = cluster_summaries(
            all_clusters, self.raw_pos, height, batch
        )
        ctype = cluster_type if cluster_type is not None else None
        pairs = build_merge_pairs(centroids, cbatch, ctype)

        if self.training:
            self._merge_cache["feat"] = None
            self._merge_cache["labels"] = None
            if pairs:
                feat = pair_feature_tensor(
                    centroids, h_mean, h_max, h_min, xy_min, xy_max, sizes, pairs
                )
                majority = cluster_majority_labels(
                    all_clusters, self.input.instance_labels.to(device)
                )
                labels = pair_labels_from_gt(majority, pairs)
                self._merge_cache["feat"] = feat
                self._merge_cache["labels"] = labels
            return all_clusters, cluster_type

        if not pairs:
            return all_clusters, cluster_type
        feat = pair_feature_tensor(centroids, h_mean, h_max, h_min, xy_min, xy_max, sizes, pairs)
        with torch.no_grad():
            probs = torch.sigmoid(self.merge_module(feat))
        merge_mask = probs > self.merge_threshold
        return union_find_merge(all_clusters, ctype, pairs, merge_mask)

    # ══════════════════════════════════════════════
    #  Scoring  (same as PointGroup3heads)
    # ══════════════════════════════════════════════

    def _compute_score(self, epoch, all_clusters, backbone_features, semantic_logits):
        mask_scores = None
        if self._scorer_type:
            x, coords, batch, pos = [], [], [], []
            for i, cluster in enumerate(all_clusters):
                x.append(backbone_features[cluster])
                coords.append(self.input.coords[cluster])
                batch.append(i * torch.ones(cluster.shape[0]))
                pos.append(self.input.pos[cluster])
            x_all = torch.cat(x)
            # ★ 2026-11-19 NaN guard: a non-finite feature matrix reaching the
            # Minkowski scorer surfaces as `cudaErrorIllegalAddress` (a sticky
            # CUDA-context fault) rather than a Python error, so it must be
            # rejected here — before the sparse tensor is built.
            if x_all.numel() == 0 or not torch.isfinite(x_all).all():
                logging.getLogger(__name__).warning(
                    "[NaNGuard] scorer input non-finite/empty (x.numel=%d) -> score skipped",
                    x_all.numel())
                return None, None
            batch_cluster = Data(
                x=x_all, coords=torch.cat(coords), batch=torch.cat(batch),
            )
            if self._voxelizer:
                batch_cluster.pos = torch.cat(pos)
                batch_cluster = batch_cluster.to(self.device)
                batch_cluster = self._voxelizer(batch_cluster)
            batch_cluster = batch_cluster.to("cpu")
            if self._scorer_type == "MLP":
                score_backbone_out = self.ScorerMLP(batch_cluster.x.to(self.device))
                cluster_feats = scatter(
                    score_backbone_out, batch_cluster.batch.long().to(self.device),
                    dim=0, reduce="max",
                )
            elif self._scorer_type == "encoder":
                score_backbone_out = self.ScorerEncoder(batch_cluster)
                cluster_feats = score_backbone_out.x
            else:
                score_backbone_out = self.ScorerUnet(batch_cluster)
                if self.mask_supervise:
                    mask_scores = self.MaskScore(score_backbone_out.x)
                    if (self.use_mask_filter_score_feature
                            and epoch > self.use_mask_filter_score_feature_start_epoch):
                        mask_index_select = torch.ones_like(mask_scores)
                        mask_index_select[
                            torch.sigmoid(mask_scores) < self.mask_filter_score_feature_thre
                        ] = 0.
                        score_backbone_out.x = score_backbone_out.x * mask_index_select
                cluster_feats = scatter(
                    score_backbone_out.x, batch_cluster.batch.long().to(self.device),
                    dim=0, reduce="max",
                )
            cluster_scores = self.ScorerHead(cluster_feats).squeeze(-1)
        else:
            with torch.no_grad():
                cluster_semantic, batch = [], []
                for i, cluster in enumerate(all_clusters):
                    cluster_semantic.append(semantic_logits[cluster, :])
                    batch.append(i * torch.ones(cluster.shape[0]))
                cluster_semantic = torch.cat(cluster_semantic)
                batch = torch.cat(batch)
                cluster_semantic = scatter(
                    cluster_semantic, batch.long().to(self.device), dim=0, reduce="mean"
                )
                cluster_scores = torch.max(torch.exp(cluster_semantic), 1)[0]
        return cluster_scores, mask_scores

    # ══════════════════════════════════════════════
    #  Loss + backward  (same as PointGroup3heads)
    # ══════════════════════════════════════════════

    def _acgo_record(self, task, weight, loss):
        """把一处 `self.loss += w·L` 同时记进 ACGO 任务表（§②：必须是**已加权**损失）。

        全记、不在这里筛，是故意的：ACGO 会**跳过** `self.loss.backward()` 自己合成
        梯度，漏记任何一项都等于静默改了目标函数。记录的是 `w·L` 而非 `L` —— 用未
        加权的 `L` 求梯度会让 offset head 的有效信号被放大 10 倍（0.1 vs 1），而日志
        上完全看不出来。
        筛选（白名单 / requires_grad / 零权重）统一在 `_acgo_backward` 里做。
        """
        if not getattr(self, "use_acgo", False):
            return
        w = float(weight)
        if task in self._acgo_tasks:
            self._acgo_tasks[task] = self._acgo_tasks[task] + w * loss
            self._acgo_nominal[task] += w
        else:
            self._acgo_tasks[task] = w * loss
            self._acgo_nominal[task] = w

    def _compute_loss(self, epoch):
        if getattr(self, "use_acgo", False):
            self._acgo_tasks = OrderedDict()
            self._acgo_nominal = OrderedDict()
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, self.labels.y.to(torch.int64), ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss
        self._acgo_record("semantic", self.opt.loss_weights.semantic, self.semantic_loss)

        self.input.instance_mask = self.input.instance_mask.to(self.device)
        self.input.vote_label = self.input.vote_label.to(self.device)
        offset_losses = offset_loss(
            self.output.offset_logits[self.input.instance_mask],
            self.input.vote_label[self.input.instance_mask],
            torch.sum(self.input.instance_mask),
        )
        for loss_name, loss in offset_losses.items():
            setattr(self, loss_name, loss)
            self.loss += self.opt.loss_weights[loss_name] * loss
            # 两个 offset 子损失并入同一个 `offset` 任务（共用 Offset head、监督同一
            # 物理量的两面；拆开会把同一目标内部的噪声算成任务间冲突）。
            self._acgo_record("offset", self.opt.loss_weights[loss_name], loss)

        self.input.instance_labels = self.input.instance_labels.to(self.device)
        self.input.batch = self.input.batch.to(self.device)
        discriminative_losses = discriminative_loss(
            self.output.embed_logits[self.input.instance_mask],
            self.input.instance_labels[self.input.instance_mask],
            self.input.batch[self.input.instance_mask].to(self.device),
            self.opt.embed_dim,
        )
        for loss_name, loss in discriminative_losses.items():
            setattr(self, loss_name, loss)
            if loss_name == "ins_loss":
                self.loss += self.opt.loss_weights.embedding_loss * loss
                self._acgo_record("embed", self.opt.loss_weights.embedding_loss, loss)

        mask_scores_sigmoid = (
            torch.sigmoid(self.output.mask_scores).squeeze()
            if self.output.mask_scores is not None else None
        )

        # ★ 2026-11-19: `cluster_scores` is None when the scorer was skipped for a
        # bad batch (see forward). `instance_ious` would crash on None.
        if (epoch > self.opt.prepare_epoch and self.use_score_net
                and self.output.cluster_scores is not None):
            ious = instance_ious(
                self.output.clusters,
                self.output.cluster_scores,
                self.input.instance_labels,
                self.input.batch,
                mask_scores_sigmoid,
                cal_iou_based_on_mask=(
                    self.cal_iou_based_on_mask
                    and epoch > self.cal_iou_based_on_mask_start_epoch
                ),
            )
        if self.output.cluster_scores is not None and self._scorer_type:
            self.score_loss = instance_iou_loss(
                ious,
                self.output.clusters,
                self.output.cluster_scores,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
                min_iou_threshold=self.opt.min_iou_threshold,
                max_iou_threshold=self.opt.max_iou_threshold,
            )
            self.loss += self.score_loss * self.opt.loss_weights["score_loss"]
            self._acgo_record("score", self.opt.loss_weights["score_loss"], self.score_loss)

        if self.use_merge and self._merge_cache.get("feat") is not None:
            merge_logits = self.merge_module(self._merge_cache["feat"])
            labels = self._merge_cache["labels"]
            # Pairs are imbalanced (most are "different tree"); without pos_weight
            # the MLP minimises BCE by predicting all-negative -> no merges (no-op).
            n_pos = labels.sum().clamp(min=1.0)
            n_neg = (1.0 - labels).sum().clamp(min=1.0)
            pos_weight = n_neg / n_pos
            self.merge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                merge_logits, labels, pos_weight=pos_weight
            )
            self.loss += self.opt.loss_weights.get("merge_loss", 1.0) * self.merge_loss
            self._acgo_record("merge", self.opt.loss_weights.get("merge_loss", 1.0), self.merge_loss)
            self._merge_cache = {}

        if self.output.mask_scores is not None and self.mask_supervise:
            self.mask_loss = mask_loss(
                ious,
                self.output.clusters,
                mask_scores_sigmoid,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
            )
            self.loss += self.mask_loss * self.opt.loss_weights["mask_loss"]
            self._acgo_record("mask", self.opt.loss_weights["mask_loss"], self.mask_loss)

        # ---- Growth consistency loss ----
        if self.use_growth_attention and self._growth_cache:
            total_mono = torch.tensor(0.0, device=self.device)
            total_prior = torch.tensor(0.0, device=self.device)
            total_attn = torch.tensor(0.0, device=self.device)
            total_corr = torch.tensor(0.0, device=self.device)
            n_corr = 0
            n_b = 0
            for b_key, cache in self._growth_cache.items():
                p = cache["p"]
                r = cache["r"]
                attn_3d2d = cache.get("attn_3d2d")
                attn_2d3d = cache.get("attn_2d3d")
                corr_3d2d = cache.get("corr_3d2d")
                corr_2d3d = cache.get("corr_2d3d")
                sp_ids = cache["sp_ids"]

                # Volume proxy: log(1 + n_points per superpoint)
                volumes = scatter(
                    torch.ones(sp_ids.shape[0], device=self.device),
                    sp_ids, dim=0, reduce='sum',
                )
                volumes = torch.log1p(volumes.float())

                losses = self.growth_loss_fn(p, r, volumes, attn_3d2d, attn_2d3d)
                total_mono = total_mono + losses["mono"]
                total_prior = total_prior + losses["prior"]
                total_attn = total_attn + losses["attn"]
                n_b += 1

                # Attention-growth correlation (diagnostic, not a loss)
                for c in (corr_3d2d, corr_2d3d):
                    if c is not None and c.numel() > 0:
                        total_corr = total_corr + c
                        n_corr += 1

            if n_b > 0:
                self.growth_mono = total_mono / n_b
                self.growth_prior = total_prior / n_b
                self.growth_attn = total_attn / n_b
                self.growth_total = self.growth_mono + self.growth_prior + self.growth_attn
                self.loss += self.growth_loss_weight * self.growth_total
                self._acgo_record("growth", self.growth_loss_weight, self.growth_total)
            self.growth_attn_corr = total_corr / n_corr if n_corr > 0 else torch.tensor(0.0, device=self.device)
        else:
            self.growth_total = torch.tensor(0.0, device=self.device)
            self.growth_mono = torch.tensor(0.0, device=self.device)
            self.growth_prior = torch.tensor(0.0, device=self.device)
            self.growth_attn = torch.tensor(0.0, device=self.device)
            self.growth_attn_corr = torch.tensor(0.0, device=self.device)

        # Clear growth cache after each forward
        self._growth_cache = {}

        # ---- ALFE diagnostics (scalar metrics, NOT losses) ----
        if self.use_allometric and self._alfe_cache:
            s_vals = torch.stack([c["s"] for c in self._alfe_cache.values()])
            self.alfe_s = s_vals.mean()
            # Cross-sample std of s_adapt: >0 means different samples get
            # different scalings (ALFE is learning sample-specific evolution).
            # std ≈ 0 means ALFE outputs the same scalar for every sample
            # (degenerate — no personalisation).
            self.alfe_s_std = s_vals.std(unbiased=False) if s_vals.numel() > 1 else torch.tensor(0.0, device=self.device)
            p_ents = []
            for c in self._alfe_cache.values():
                p = c["p"]
                p_ents.append(-(p * torch.log(p + 1e-8)).sum())
            self.alfe_p_entropy = torch.stack(p_ents).mean()
        else:
            self.alfe_s = torch.tensor(0.0, device=self.device)
            self.alfe_s_std = torch.tensor(0.0, device=self.device)
            self.alfe_p_entropy = torch.tensor(0.0, device=self.device)
        self._alfe_cache = {}

        # ---- Causal growth attention: sparsity LOSS + diagnostics ----
        # L_sparse = λ_s · mean(A) 是 SPARTAN Eq.6 的稀疏正则，是"因果结构"
        # 的训练信号（与去混淆的纯诊断指标不同，这个必须进 loss）。它惩罚
        # 活跃因果边数，迫使信息流走稀疏的因果链路，而不是稠密 MLP。
        if self.use_causal_growth:
            edge_probs = []
            token_probs = []
            for c in self._causal_growth_cache.values():
                if c["edge_prob"].numel() > 0:
                    edge_probs.append(c["edge_prob"])
                if c["token_prob"].numel() > 0:
                    token_probs.append(c["token_prob"])
            if edge_probs:
                self.causal_edge_sparsity = torch.cat(edge_probs).mean()
            else:
                self.causal_edge_sparsity = torch.tensor(0.0, device=self.device)
            if token_probs:
                self.causal_token_sparsity = torch.cat(token_probs).mean()
            else:
                self.causal_token_sparsity = torch.tensor(0.0, device=self.device)
            # 平均边稀疏度 + token 稀疏度作为总稀疏度量
            self.causal_sparse = 0.5 * (self.causal_edge_sparsity + self.causal_token_sparsity)
            self.loss += self.causal_sparse_weight * self.causal_sparse
            # ★ 2026-09-25 新增：把"稀疏正则施加在门偏置上的梯度"解析地记下来，
            #   便于在 backward 里把 `edge_logit_bias.grad` **精确分解**成
            #   「任务梯度」+「稀疏梯度」，从而一次定下 λ_s，而不是扫一串值。
            #   A = σ(z + bias) ⇒ dA/dbias = A(1−A)；
            #   causal_sparse = 0.5·(mean(A) + mean(token_prob))，而 token_prob 是
            #   σ(score_g) 与 bias 无关 ⇒ d(causal_sparse)/d(bias) = 0.5·mean(A(1−A))。
            if edge_probs:
                _allA = torch.cat(edge_probs)
                self.causal_gate_sensitivity = 0.5 * (_allA * (1.0 - _allA)).mean()
                self.causal_gate_spread = _allA.std().detach()
            else:
                self.causal_gate_sensitivity = torch.tensor(0.0, device=self.device)
                self.causal_gate_spread = torch.tensor(0.0, device=self.device)
            # 零权重项：记进去由 `_acgo_backward` 的零权重守卫排除（naive 实现直接
            # 求梯度会拿到恒 0 梯度，进 cosine 就是纯噪声）。
            self._acgo_record("causal_sparse", self.causal_sparse_weight, self.causal_sparse)
        self._causal_growth_cache = {}

        # ---- Causal deconfounding: keep-ratio + decorrelation DIAGNOSTICS ----
        # PURELY diagnostic — neither term is added to the loss. causal_keep_ratio
        # = fraction of 2D-feature energy retained after the backdoor projection
        # (1.0 = no-op); causal_decorr = mean squared cosine between the
        # deconfounded residual and the 3D feature.
        #
        # The original version ADDED causal_decorr to the loss, which created a
        # dead-zone: src_2d and dst_3d are independent modality streams (different
        # proj_2d/proj_3d + gru), so they start near-orthogonal, the decorrelation
        # objective is already satisfied (decorr ≈ 0.004 < random 1/D ≈ 0.0078) and
        # contributes ≈0 gradient. The confounder is now trained by the MAIN loss
        # alone, through the low-rank NWGM projection in forward() (Causal-SETR
        # mechanism).
        if self.use_causal:
            if self._causal_cache:
                keeps = []
                decorrs = []
                for c in self._causal_cache.values():
                    keeps.append(c["keep"])
                    s = c["src2d"].detach()              # [E, D_msg] — diagnostic only
                    d = c["dst3d"].detach()              # [E, D_msg]
                    w = torch.softmax(
                        s @ self.confounder_dict.T / (self.D_msg ** 0.5), dim=-1
                    )
                    resid = s - w @ self.confounder_dict  # [E, D_msg]
                    rn = resid / (resid.norm(dim=-1, keepdim=True) + 1e-8)
                    dn = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
                    decorrs.append(((rn * dn).sum(-1) ** 2).mean())
                self.causal_keep_ratio = torch.stack(keeps).mean()
                self.causal_decorr = torch.stack(decorrs).mean()
            else:
                self.causal_keep_ratio = torch.tensor(1.0, device=self.device)
                self.causal_decorr = torch.tensor(0.0, device=self.device)
        self._causal_cache = {}

    # ──────────────────────────────────────────────
    #  ACGO backward (S3)
    # ──────────────────────────────────────────────

    def _acgo_active(self, epoch):
        return getattr(self, "use_acgo", False) and epoch >= self.acgo_start_epoch

    def _acgo_backward(self, epoch):
        """用 ACGO 合成多任务梯度并写回 `p.grad`（不调 `self.loss.backward()`）。

        之后的 `optimize_parameters2` 原样继续（`_do_unscale_loss` 在
        `is_mixed_precision()` 分支内且本模型恒为 False ⇒ `p.grad` 会原样活着走到
        `optimizer.step()`）。完全不动 `base_model.py`。
        """
        from .acgo import acgo_combine, flatten, split_params, write_grads

        assert not self.is_mixed_precision(), (
            "[ACGO] mixed precision not supported: _do_unscale_loss/GradScaler would "
            "rescale p.grad AFTER ACGO wrote it, and that path was never validated for "
            "manually-written gradients."
        )

        shared, private = split_params(self.named_parameters(), self.acgo_shared_prefixes)
        if not shared or not private:
            logging.getLogger(__name__).warning(
                "[ACGO] unexpected split (shared=%d, private=%d) — falling back to "
                "self.loss.backward()" % (len(shared), len(private))
            )
            self.loss.backward()
            return

        # ---- 任务筛选（§④：白名单 + requires_grad + 零权重，三重守卫）----
        # 白名单是**显式**的，不从 loss_names 推导：loss_names 是"要记录的诊断标量"
        # 的列表，两者语义不同。
        tasks, dropped = [], []
        for name in self._acgo_task_names:
            if name not in self._acgo_tasks:
                dropped.append(f"{name}:absent")
                continue
            L = self._acgo_tasks[name]
            if not L.requires_grad:
                # 死项：如 freeze_backbone 下 semantic_logits 由 no_grad 产出、
                # num_instances==0 时退化成 torch.tensor(0.) 的 ins_loss。
                dropped.append(f"{name}:no_grad_fn")
                continue
            if self._acgo_nominal.get(name, 0.0) == 0.0:
                dropped.append(f"{name}:w=0")
                continue
            tasks.append((name, L))

        if len(tasks) < 2:
            logging.getLogger(__name__).warning(
                "[ACGO] only %d usable task(s) %s — nothing to project; falling back "
                "to self.loss.backward()"
                % (len(tasks), [n for n, _ in tasks])
            )
            self.loss.backward()
            return

        all_named = shared + private
        all_p = [p for _, p in all_named]
        all_names = [n for n, _ in all_named]
        shared_names = set(n for n, _ in shared)

        gs, gp = [], []
        counts = [0] * len(all_named)
        shared_p = [p for _, p in shared]
        private_p = [p for _, p in private]

        # ---- ★ 修正 (2026-09-25)：把"非 ACGO 任务"的损失梯度补回来 ----
        # 本方法**跳过 `self.loss.backward()`**，而 `_acgo_task_names` 只有 3 个
        # （offset/embed/score）。于是 `_acgo_record` 记下的另外几项
        # （semantic / growth / merge / mask / causal_sparse）梯度被**静默丢弃**：
        #   - 不在 tasks 里 ⇒ 从不参与求梯度；
        #   - 只被它们触达的参数（最典型 = 整个 `Semantic.*` 头）触达计数 0
        #     ⇒ `write_grads(seen=False)` 跳过不写 ⇒ `p.grad` 保持 None
        #     ⇒ `optimizer.step()` **不更新它们**。
        # 这等于一开 ACGO 就静默换了目标函数 —— 正是 `_acgo_record` 的 docstring
        # 警告过的情形，而且日志看不出来：`dropped=` 那行的循环只遍历白名单，
        # 永远不会打印 `semantic:dropped`。
        #
        # 修法：`rest = self.loss − Σ(3 个任务的加权项)`，单独求一次梯度叠加回去。
        # 这个减法是**精确的**（不是近似）：`_acgo_record(name, w, L)` 在全文件里
        # 永远紧跟在 `self.loss += w*L` 之后、且传同一个 `w` ⇒ self.loss 与
        # _acgo_tasks 只是同一批乘积的两种分组。
        task_sum = None
        for _n in self._acgo_task_names:
            _L = self._acgo_tasks.get(_n)
            if _L is not None:
                task_sum = _L if task_sum is None else task_sum + _L
        rest = self.loss if task_sum is None else (self.loss - task_sum)
        # `self.loss` 可能退化成 Python float（所有项权重为 0 的极端情形），
        # 所以不直接取 `.requires_grad`。
        rest_requires_grad = getattr(rest, "requires_grad", False)
        rest_names = [n for n in self._acgo_tasks
                      if n not in [t for t, _ in tasks]]

        for i, (_, L) in enumerate(tasks):
            g = torch.autograd.grad(
                L, all_p,
                # 最后一个任务负责释放图；但若下面还要对 `rest` 求一次梯度，
                # 这里必须继续保留（否则 rest 那次会踩到已释放的图）。
                retain_graph=(i < len(tasks) - 1) or rest_requires_grad,
                allow_unused=True,
            )
            # 触达计数就地累加，**不**保留整套 per-task 梯度：3 个任务 × 全部可训
            # 参数的梯度同时驻留会白占几十 MB 的显存，而本 run 是 40h 级的。
            for k, gi in enumerate(g):
                if gi is not None:
                    counts[k] += 1
            # `materialize_grads` 在 torch 1.9 不存在，手写 None → zeros
            g = [torch.zeros_like(p) if gi is None else gi for gi, p in zip(g, all_p)]
            gs.append(flatten(g[:len(shared)], shared_p))
            gp.append(flatten(g[len(shared):], private_p))
        g_rest = None
        if rest_requires_grad:
            # 这一次用 retain_graph=False ⇒ 由它负责释放图（上面已据此保留）。
            g_rest = torch.autograd.grad(rest, all_p, allow_unused=True)

        nominal = [self._acgo_nominal[n] for n, _ in tasks]
        if getattr(self, "acgo_variant", "acgo") == "gradnorm":
            # ★ 2026-10-24：GradNorm 对照臂。不走冲突检测/投影，只按梯度幅值
            #   平衡权重。`_acgo_record` 存的是 **w·L**，这里除回标称权重拿未加权 L。
            from .acgo import gradnorm_combine
            task_losses = [
                (self._acgo_tasks[n] / max(float(self._acgo_nominal[n]), 1e-12)).detach()
                for n, _ in tasks
            ]
            g_shared, g_private, diag = gradnorm_combine(
                gs, gp, task_losses, self._gradnorm_state,
                alpha=self.gradnorm_alpha, lr=self.gradnorm_lr, eps=self.acgo_eps,
            )
            # 下游诊断代码（上面那段 setattr / 日志行）期望 acgo_* 这套键。
            # GradNorm 没有冲突概念，补 0 / 把 w 映射到 mod 位置，保持同一份日志骨架。
            diag["acgo_num_tasks"] = float(len(tasks))
            diag["acgo_conflict_ratio"] = 0.0
            diag["acgo_mean_cosine"] = 0.0
            for _i, _ in enumerate(tasks):
                diag[f"acgo_conflict_task{_i}"] = 0.0
                diag[f"acgo_gnorm_task{_i}"] = diag.get(f"gradnorm_G{_i}", 0.0)
                diag[f"acgo_mod_task{_i}"] = diag.get(f"gradnorm_w{_i}", 1.0)
        else:
            g_shared, g_private, diag = acgo_combine(
                gs, gp, nominal, self._acgo_state,
                projection=self.acgo_projection,
                adaptive=self.acgo_adaptive_weight,
                lam=self.acgo_lambda,
                beta=self.acgo_ema_beta,
                eps=self.acgo_eps,
                conflict_threshold=self.acgo_conflict_threshold,
            )

        # 把 rest 的梯度加到合成结果上：ACGO 只负责重新分配**任务间**的梯度，
        # 其余监督信号应原样通过（与 `self.loss.backward()` 下的行为一致）。
        # 坐标不冲突：3 个任务全为 None 的坐标在 acgo_combine 里已是 0，
        # 加上 rest 就等于 rest 本身。
        if g_rest is not None:
            g_rest = [torch.zeros_like(p) if gi is None else gi
                      for gi, p in zip(g_rest, all_p)]
            g_shared = g_shared + flatten(g_rest[:len(shared)], shared_p)
            g_private = g_private + flatten(g_rest[len(shared):], private_p)
            seen_all = [(counts[k] > 0) or (g_rest[k] is not None)
                        for k in range(len(all_named))]
        else:
            seen_all = [c > 0 for c in counts]
        self._acgo_rest_value = (float(rest.detach()) if rest_requires_grad else 0.0)
        self._acgo_rest_names = rest_names

        n_ws = write_grads(shared, g_shared, seen=seen_all[:len(shared)])
        n_wp = write_grads(private, g_private, seen=seen_all[len(shared):])
        if n_ws + n_wp == 0:
            # 静默跳过 optimizer step 是本项目最贵的失败模式（head 冻死 119 轮且
            # 日志看不出来），所以这里宁可炸。
            raise RuntimeError(
                "[ACGO] wrote 0 gradients — task set is disconnected from every "
                "trainable parameter. Refusing to continue."
            )

        # ---- 诊断进 self.<name>（已在 loss_names 里 ⇒ 自动进 wandb）----
        for i, (name, _) in enumerate(tasks):
            for kind in ("conflict", "gnorm", "mod"):
                setattr(self, f"acgo_{kind}_{name}",
                        diag.get(f"acgo_{kind}_task{i}", 0.0))
        for name in self._acgo_task_names:
            if name not in [n for n, _ in tasks]:
                for kind in ("conflict", "gnorm", "mod"):
                    setattr(self, f"acgo_{kind}_{name}", 0.0)
        self.acgo_num_tasks = diag["acgo_num_tasks"]
        self.acgo_conflict_ratio = diag["acgo_conflict_ratio"]
        self.acgo_mean_cosine = diag["acgo_mean_cosine"]
        self.acgo_shared_flat_dim = float(g_shared.numel())

        # ---- 每轮一次的可审计日志（ACGO 会跳过 self.loss.backward()，必须能看出
        #      到底哪些损失进了合成、哪些被丢了）----
        if getattr(self, "_acgo_last_logged_epoch", -1) != epoch:
            lg = logging.getLogger(__name__)
            lg.info(
                "[ACGO][ep%d] tasks=%s nominal_w=%s dropped=%s "
                "shared=%d params / %d dim | private=%d params | "
                "conflict_ratio=%.4f mean_cos=%+.4f mod=%s conflict=%s gnorm=%s "
                "wrote=(s%d,p%d) | rest=%.4f via=%s"
                % (epoch, [n for n, _ in tasks], [round(w, 4) for w in nominal],
                   dropped or "none", len(shared), g_shared.numel(), len(private),
                   self.acgo_conflict_ratio, self.acgo_mean_cosine,
                   [round(getattr(self, "acgo_mod_" + n), 4)
                    for n in self._acgo_task_names],
                   [round(getattr(self, "acgo_conflict_" + n), 3)
                    for n in self._acgo_task_names],
                   ["%.3e" % getattr(self, "acgo_gnorm_" + n)
                    for n in self._acgo_task_names],
                   n_ws, n_wp,
                   getattr(self, "_acgo_rest_value", 0.0),
                   self._acgo_rest_names or "none")
            )
            # 自审计①：名单外却 >=2 个任务触达 ⇒ `acgo_shared_prefixes` 漏了
            missing = [all_names[i] for i, c in enumerate(counts)
                       if c >= 2 and all_names[i] not in shared_names]
            if missing:
                lg.warning(
                    "[ACGO] acgo_shared_prefixes MISSES multi-task params (conflicts "
                    "there are NOT being detected): %s" % missing
                )
            # 自审计②：名单内却只有 1 个任务触达 ⇒ 该参数在稀释 cosine（§③）
            single = [all_names[i] for i, c in enumerate(counts[:len(shared)])
                      if c == 1]
            if single:
                lg.info("[ACGO] SHARED params touched by only 1 task (dilutes "
                        "cosine): %s" % single)
            self._acgo_last_logged_epoch = epoch

    def backward(self, epoch):
        self._compute_loss(epoch)
        # ★★ 2026-11-19 NaN guard, corrected. The 2026-11-18 guard lived in
        # `base_model.optimize_parameters2` and ran BEFORE `_compute_loss` (which is
        # called on the line above), so it only ever inspected the *previous*
        # batch's loss tensors — it never stopped the update of the poisoned batch
        # itself. NaN gradients therefore reached Adam, the weights went NaN, and
        # every later batch died downstream (the observed `cudaErrorIllegalAddress`
        # in the Minkowski scorer). This check runs on the CURRENT losses and
        # returns without calling backward at all.
        _loss_bad = torch.is_tensor(self.loss) and not torch.isfinite(self.loss).all()
        if _loss_bad or getattr(self, "_bad_batch", False):
            _named = [
                n for n in self.loss_names
                if torch.is_tensor(getattr(self, n, None))
                and not torch.isfinite(getattr(self, n)).all()
            ]
            logging.getLogger(__name__).warning(
                "[NaNGuard] ep%s: skipping optimizer step — loss_bad=%s batch_flag=%s "
                "non_finite=%s", epoch, bool(_loss_bad),
                bool(getattr(self, "_bad_batch", False)), _named or ["loss"])
            self._optimizer.zero_grad()
            return

        _acgo_on = self._acgo_active(epoch)
        if _acgo_on:
            self._acgo_backward(epoch)
        else:
            self.loss.backward()
        _acgo_tag = "[ACGO] " if _acgo_on else ""
        # ---- Causal gradient diagnostic (once per epoch) ----
        # Hard evidence for whether the confounder actually receives a non-zero
        # gradient. If grad_norm ≈ 0 while keep_ratio stays ~1.0, the module is
        # in a gradient dead-zone (weak init / empty decorr loss); if grad_norm
        # is healthy and keep_ratio drops below 1, the confounder is genuinely
        # learning a non-trivial deconfounding representation.
        if self.use_causal and getattr(self, "_causal_last_logged_epoch", -1) != epoch:
            g = self.confounder_dict.grad
            if g is not None:
                import logging
                logging.getLogger(__name__).info(
                    f"{_acgo_tag}[Causal][ep{epoch}] confounder grad_norm={g.norm().item():.6e} "
                    f"param_norm={self.confounder_dict.norm().item():.4f}"
                )
            self._causal_last_logged_epoch = epoch

        # ---- Causal growth gradient diagnostic (once per epoch) ----
        if self.use_causal_growth and getattr(self, "_cgg_last_logged_epoch", -1) != epoch:
            # ★ 2026-10-24：GAT 对照臂没有 growth_token / edge_logit_bias，
            #   原来的诊断会 AttributeError。这里按模块类型分流，日志字段不同但
            #   仍是「每轮一行、可 grep」的同一契约。
            if not hasattr(self.causal_growth_attn, "growth_token"):
                import logging as _logging
                _gwo = self.causal_growth_attn.W_o.weight.grad
                _logging.getLogger(__name__).info(
                    f"{_acgo_tag}[{self.causal_growth_attn_type.upper()}][ep{epoch}] W_o "
                    f"grad_norm={0.0 if _gwo is None else _gwo.norm().item():.6e} "
                    f"edge_sparsity={getattr(self, 'causal_edge_sparsity', float('nan')):.6f} "
                    f"gate_spread={float(getattr(self, 'causal_gate_spread', 0.0)):.6f} "
                    f"（本臂这两个量 = softmax 注意力权重的均值/标准差，"
                    f"不是门控概率，跨臂不可比）"
                )
                self._cgg_last_logged_epoch = epoch
                g = None
            else:
                g = self.causal_growth_attn.growth_token.grad
            if g is not None:
                import logging
                # ★ 2026-09-25 扩充：把 `edge_logit_bias` 上的梯度**分解**成
                #   「任务梯度」+「稀疏正则梯度」，用来判定"门是不是被模型自己中性化了"
                #   以及"λ_s 需要多大才构成有意义的少数派压力"。
                #   分解是解析的、不需要第二次 backward：
                #       g_sparse = λ_s · 0.5 · mean(A(1−A))     （见 _compute_loss）
                #       g_task   = g_total − g_sparse
                #   ⚠️ 口径限制：**ACGO 开着的臂（机5/机6）**里 p.grad 是 ACGO
                #      **合成后**的梯度（并已叠加 rest），所以那条的 g_task 是
                #      近似值、只能看量级；**机1/机3（无 ACGO）的分解是精确的**。
                #      行首的 `[ACGO]` 前缀就是标记。
                _bias_p = self.causal_growth_attn.edge_logit_bias
                _g_tot = _bias_p.grad
                _sens = float(getattr(self, "causal_gate_sensitivity", 0.0))
                _g_sp = self.causal_sparse_weight * _sens
                _g_tot_v = 0.0 if _g_tot is None else float(_g_tot)
                _g_task = _g_tot_v - _g_sp
                _ratio = abs(_g_sp) / (abs(_g_task) + 1e-12)
                logging.getLogger(__name__).info(
                    f"{_acgo_tag}[CausalGrowth][ep{epoch}] growth_token "
                    f"grad_norm={g.norm().item():.6e} "
                    f"edge_sparsity={getattr(self, 'causal_edge_sparsity', float('nan')):.4f} "
                    f"gate_spread={float(getattr(self, 'causal_gate_spread', 0.0)):.4f} "
                    f"bias={float(_bias_p):+.4f} "
                    f"g_bias_total={_g_tot_v:+.6e} g_task={_g_task:+.6e} "
                    f"g_sparse={_g_sp:+.6e} ratio={_ratio:.4f} lam_s={self.causal_sparse_weight}"
                )
            self._cgg_last_logged_epoch = epoch

        # ---- Growth feature diagnostic (once per epoch) ----
        # ★ 2026-09-25 新增。这是"生长模块是活的还是死的"的唯一直接证据：
        #   若 |W| 与 |残差| 长期停在 1e-8 量级，整条生长分支等于 no-op（而 F1 上看
        #   不出来，只表现为"加了生长和没加一样"）。零初始化**不**阻塞梯度
        #   （dL/dW = (dL/dres)ᵀ·h 与 W 无关），所以这行是用来**验证**而不是假设。
        if self.use_growth_feature and getattr(self, "_growth_proj_last_logged_epoch", -1) != epoch:
            import logging
            _w = self.growth_proj.weight.detach()
            _gw = self.growth_proj.weight.grad
            _res = getattr(self, "_growth_res_probe", None)
            logging.getLogger(__name__).info(
                f"{_acgo_tag}[GrowthFeature][ep{epoch}] |W|={_w.norm().item():.4e} "
                f"|W|max={_w.abs().max().item():.4e} "
                f"grad_norm={'None' if _gw is None else '%.4e' % _gw.norm().item()} "
                f"res_absmean={'None' if _res is None else '%.4e' % float(_res)}"
            )
            self._growth_proj_last_logged_epoch = epoch

        # ---- Weight sanity (once per epoch) ----
        # ★ 2026-11-19: if a stack of NaN ever slips past the guard above, this
        # line makes it visible within minutes instead of after a CUDA fault
        # hours later. Diagnostic only — it never modifies training.
        if getattr(self, "_w_check_epoch", -1) != epoch:
            self._w_check_epoch = epoch
            _bad_w = [n for n, p in self.named_parameters()
                      if p.requires_grad and not torch.isfinite(p.detach()).all()]
            if _bad_w:
                logging.getLogger(__name__).error(
                    "[NaNGuard] ep%s: NON-FINITE WEIGHTS in %d params (e.g. %s) "
                    "— the checkpoint is poisoned; stop and resume from an earlier one",
                    epoch, len(_bad_w), _bad_w[:5])

    # ══════════════════════════════════════════════
    #  Visualisation
    # ══════════════════════════════════════════════

    def _dump_visuals(self, epoch):
        if random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y,
                instance_labels=self.input.instance_labels, batch=self.input.batch,
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.vote = self.output.offset_logits
            nms_idx = self.output.get_instances()
            if self.output.clusters is not None:
                data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
                data_visual.cluster_type = self.output.cluster_type[nms_idx]
            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
