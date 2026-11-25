"""
Causal Growth Attention — 方案 A：因果掩码注意力 + 生长干预 token。

作为「残差」加在 Stage 1 的 2D→3D 稠密 MLP 消息传递之上（v5 注入 2D
信息的关键路径）。调用方负责 `agg_3d = dense + causal_residual`。

对齐老师四篇因果论文 + MVCTNet 生长篇的正确机制（"用因果结构约束信息
流本身"，而非"估计生长/因果信号再调制特征"）：

  - 可学习稀疏因果邻接  A_ij = σ(q_iᵀk_j)      [SPARTAN Eq.3]
      每个 superpoint 动态采样它自己的因果 parent 集（状态依赖的局部因果图），
      与"全局图 ≈ 全连接 ≈ 无信息量"相对。
  - 因果掩码            masked_score = score + log(A_ij)   [DAG-aware Eq.6]
      非因果边（A_ij≈0）的注意力被抑制，信息只沿因果链路流动。
      ★ 2026-09-25 修正：原实现是 `score · A_ij`，数学上不成立（A→0 给出
      exp(0)=1 = 完全保留，而非抑制），且会退化成均匀 softmax。详见 forward
      里"因果掩码"那段的推导。
  - 生长干预 token      可学习 token，作为所有 superpoint 的全局因果 parent
      [SPARTAN §3.2]，编码异速生长先验（MVCTNet ALFE 的 b_eff/竞争特征）。
  - 小量非零输出残差    Z = W_o(H)，W_o ≈ 0 → 起步近似 no-op  [DAG-aware Z = αH + X]
      模型从 v5 结果出发；非零 init 给边概率留可微路径（避免稀疏正则压到吸收态 0）。

与之前三次失败（生长注意力 v1-v4 / ALFE / 因果去混淆）的本质区别：
之前是"估计信号→调制特征"，这里是"采样因果邻接→约束信息流走哪条边"。
稀疏正则 L_sparse = λ_s · mean(A) 是因果结构的载体（SPARTAN Eq.6）。

设计约束：PyTorch ≥ 1.9.0，无 scaled_dot_product_attention，多头注意力用
手动 reshape 实现（与 growth_attention.py 一致）。
"""

import math
import torch
import torch.nn as nn
from torch_scatter import scatter


def gumbel_sigmoid(logits, tau=1.0, hard=False):
    """Gumbel-Sigmoid 采样（SPARTAN Eq.3 的 Bern(σ(·)) 的可微版本）。

    hard=False（默认，训练用）：直接返回确定性 sigmoid —— 即 Bern 参数的
    期望，去掉每步采样噪声，稀疏正则的梯度平滑、训练稳定。稀疏性通过
    λ_s·mean(A) 的正则项（而非硬采样）获得。

    hard=True：straight-through Gumbel-Sigmoid，得到真正的 {0,1} 邻接矩阵
    （可解释 SHD，但训练更敏感、易 mode collapse，SPARTAN 原文也承认）。
    """
    if hard:
        g1 = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8) + 1e-8) + 1e-8)
        g0 = -torch.log(-torch.log((1.0 - torch.rand_like(logits)).clamp(min=1e-8) + 1e-8) + 1e-8)
        y_soft = torch.sigmoid((logits + g1 - g0) / tau)
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return torch.sigmoid(logits)


class CausalGrowthAttention(nn.Module):
    """稀疏因果邻接 + 生长干预 token 的跨模态注意力（2D patch → 3D superpoint）。

    Args:
        D_msg:     消息维度（128）
        num_heads: 注意力头数
        tau:       Gumbel-Sigmoid 温度（仅 hard=True 时生效）
        use_hard_gumbel: 是否用 straight-through 硬采样（默认 False，软稀疏）

    Input:
        src_2d:  [E, D_msg]  2D patch 特征（cause）
        dst_3d:  [E, D_msg]  边上的 superpoint 特征（effect）
        sp_feat: [M_dst, D_msg] 完整 superpoint 特征（生长 token 的 query）
        edges:   [E, 2]      (patch_idx, superpoint_idx)
        M_dst:   int         superpoint 总数

    Output:
        out:        [M_dst, D_msg] 聚合消息（调用方负责 gru/residual）
        edge_prob:  [E]            因果边概率 A_ij（稀疏正则 + 诊断）
        token_prob: [M_dst]        生长 token 因果概率（稀疏正则 + 诊断）
    """

    def __init__(self, D_msg=128, num_heads=4, tau=1.0, use_hard_gumbel=False,
                 edge_prob_floor=0.05):
        super().__init__()
        assert D_msg % num_heads == 0, (
            f"D_msg ({D_msg}) must be divisible by num_heads ({num_heads})"
        )
        assert 0.0 < edge_prob_floor < 1.0, (
            f"edge_prob_floor must be in (0, 1), got {edge_prob_floor}"
        )
        self.D_msg = D_msg
        self.num_heads = num_heads
        self.d_k = D_msg // num_heads
        self.tau = tau
        self.use_hard_gumbel = use_hard_gumbel
        # log 域因果门控的下限（见 forward 里"因果掩码"那段）：防止 log(0) = -inf
        # 引发 NaN，同时挡住"边概率全部归零 ⇒ 模块无梯度"的吸收态。
        self.edge_prob_floor = edge_prob_floor

        # Query (superpoint / effect), Key / Value (patch / cause)
        self.W_q = nn.Linear(D_msg, D_msg, bias=False)
        self.W_k = nn.Linear(D_msg, D_msg, bias=False)
        self.W_v = nn.Linear(D_msg, D_msg, bias=False)

        # 生长干预 token（可学习，编码异速生长先验）
        self.growth_token = nn.Parameter(torch.randn(1, D_msg) * 0.1)

        # 稀疏先验（CausalNET 教训：因果图负/零初始化远好于正初始化）。
        # 边 logits 加负偏置 → edge_prob = σ(score + bias) 起步 < 0.5（稀疏），
        # 而非半稠密 0.5。可学习，主任务梯度可按需移动它。
        self.edge_logit_bias = nn.Parameter(torch.tensor(-2.0))

        # 输出投影：小量非零初始化（原 zero-init 挡住 edge_prob/token_prob 的主任务
        # 梯度，配合 λ·mean(A) 稀疏正则把边压到吸收态 0 → 注意力死）。std 0.01 的
        # 残差仍近似 no-op，但从 epoch 0 就给了边概率可微路径；无用时主损失会压回 0。
        self.W_o = nn.Linear(D_msg, D_msg, bias=False)
        nn.init.normal_(self.W_o.weight, mean=0.0, std=0.01)
        self.W_og = nn.Linear(D_msg, D_msg, bias=False)
        nn.init.normal_(self.W_og.weight, mean=0.0, std=0.01)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, src_2d, dst_3d, sp_feat, edges, M_dst):
        device = src_2d.device
        if edges.shape[0] == 0:
            return (
                torch.zeros(M_dst, self.D_msg, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            )

        src_idx = edges[:, 0]
        dst_idx = edges[:, 1]
        E = src_idx.shape[0]

        # ---- Q, K, V（多头）----
        Q = self.W_q(dst_3d).view(E, self.num_heads, self.d_k)   # [E, H, d_k]
        K = self.W_k(src_2d).view(E, self.num_heads, self.d_k)   # [E, H, d_k]
        V = self.W_v(src_2d).view(E, self.num_heads, self.d_k)   # [E, H, d_k]

        # ---- 注意力分数 ----
        score = (Q * K).sum(dim=-1) / math.sqrt(self.d_k)        # [E, H]

        # ---- 可学习稀疏因果邻接（SPARTAN Eq.3）----
        # 边级因果概率 = head 平均分数的 sigmoid（Bernoulli 参数）
        edge_prob = gumbel_sigmoid(
            score.mean(dim=-1) + self.edge_logit_bias, self.tau, self.use_hard_gumbel
        )                                                        # [E] ∈ [0,1]

        # ---- 因果掩码（DAG-aware Eq.6 的连续化）----
        # ★ 修正 (2026-09-25)：原实现写的是 `score * edge_prob`，**数学上不成立**。
        #   Bernoulli 门控 A_ij 是乘在 softmax **权重**上的：attn_ij ∝ A_ij·exp(score_ij)。
        #   要变成"指数里加一项"必须取对数：attn ∝ exp(score + log A)。
        #   用 `score * A` 的实际行为（sigmoid 后 A∈[0,1]）：
        #       A = 1 → exp(score)   ✓ 与意图一致
        #       A = 0 → exp(0) = 1   ✗ **完全保留**，而意图是"完全抑制"
        #   更致命的是：A→0 恰好是稀疏正则 λ·mean(A)（SPARTAN Eq.6）的**最优解**
        #   ⇒ 训练把所有边概率一起压向 0 ⇒ masked_score → 0 对所有边
        #   ⇒ softmax 退化成**均匀分布** ⇒ 模块等价于邻域 mean-pooling，
        #   与它要细化的 `agg_dense`（稠密 MLP 聚合）功能重复 ⇒ 因果结构失效、Δ=0。
        #   这解释了历史记录里的"因果注意力已死 / 各次因果模块均为负结果"。
        #
        #   floor 用**加性**平滑（log(A + floor)），不用 clamp：clamp 在 A < floor
        #   时梯度恰好为 0 ⇒ 掉到 floor 以下的边永远回不来，等于把吸收态从 A=0
        #   挪到了 A=floor。加性形式处处可导：∂log(A+floor)/∂A = 1/(A+floor) > 0，
        #   且 A→0 时给出有限的 log(floor)（不产生 -inf ⇒ 下面减 score_max 不会
        #   出现 -inf-(-inf)=NaN，这正是当初绕开 log 域、改用乘法然后写错的根因）。
        #   floor 的量级决定抑制深度：A=1 与 A=0 的 log 差 = log((1+floor)/floor)，
        #   floor=0.05 → 3.0 个自然对数（≈ 到 5% 的权重），floor=0.01 → 4.6。
        masked_score = (score
                        + torch.log(edge_prob + self.edge_prob_floor).unsqueeze(-1))  # [E, H]

        # ---- softmax over incoming edges per dst ----
        score_max = scatter(masked_score, dst_idx, dim=0, reduce='max',
                            dim_size=M_dst)                       # [M_dst, H]
        score_shifted = masked_score - score_max[dst_idx]
        score_exp = torch.exp(score_shifted)
        Z = scatter(score_exp, dst_idx, dim=0, reduce='sum',
                    dim_size=M_dst)                               # [M_dst, H]
        attn = score_exp / Z[dst_idx].clamp(min=1e-8)             # [E, H]

        # ---- 加权聚合 ----
        msg = attn.unsqueeze(-1) * V                              # [E, H, d_k]
        msg = msg.view(E, self.D_msg)
        out = scatter(self.W_o(msg), dst_idx, dim=0, reduce='sum',
                      dim_size=M_dst)                             # [M_dst, D]

        # ---- 生长干预 token（SPARTAN §3.2）----
        # 每个 superpoint 对生长 token 的因果概率（全局因果 parent）
        Q_sp = self.W_q(sp_feat).view(M_dst, self.num_heads, self.d_k)  # [M_dst, H, d_k]
        K_g = self.W_k(self.growth_token).view(1, self.num_heads, self.d_k)
        score_g = (Q_sp * K_g).sum(dim=-1) / math.sqrt(self.d_k)  # [M_dst, H]
        token_prob = gumbel_sigmoid(
            score_g.mean(dim=-1), self.tau, self.use_hard_gumbel
        )                                                        # [M_dst] ∈ [0,1]

        token_val = self.W_v(self.growth_token)                   # [1, D]
        g_out = token_prob.unsqueeze(-1) * token_val              # [M_dst, D]
        out = out + self.W_og(g_out)                              # [M_dst, D]

        return out, edge_prob, token_prob
