"""
ACGO — Adaptive Conflict-Guided Optimisation（S3 的优化算法臂）。

多任务梯度：检测 pairwise 冲突 → PCGrad 式投影 → 按近期冲突自适应调制权重
→ 写回 `p.grad`，交回原 Adam。**不换优化器、不新增网络模块。**

本文件是**纯函数**集合：不持有模型/优化器状态，EMA 状态由调用方传入传出，
可脱离 torch_points3d 独立单测（见 test_acgo.py）。

── 设计约束（来自 plan `parallel-doodling-abelson.md` §一，逐条对应实现）──

① 参数分 SHARED / PRIVATE 两类，**不是**只取共享：
   - SHARED（被 ≥2 个任务触达的融合路径）→ 冲突检测 + 投影
   - PRIVATE（只被 1 个任务触达的 head / scorer）→ 直接求和写回
   若只写共享参数的 grad 而跳过 `self.loss.backward()`，三个 head 的 `.grad`
   会是 None，而 Adam 对 `grad is None` 不报错直接跳过 —— 119 轮里 head 一步
   不动且日志看不出来。

② 任务梯度必须是**已加权**的 `w·L`。用未加权的 `L` 求梯度会让 offset head 的
   有效信号被放大 10 倍（权重 0.1 vs 1），日志上完全看不出来。

③ 冲突检测与投影**只在 SHARED 子空间内做**。把各任务的"共享+私有"梯度拼成
   一个大向量、缺失位置填 0，会让 cosine 被私有块的质量稀释，而且 PCGrad 的
   修正项 `−(g_i·g_j/‖g_j‖²)·g_j` 会作用到 `g_i` 的**私有坐标**上，把 `g_j`
   的私有梯度注入进来 —— 恰好制造了 PCGrad 本该消除的跨任务污染。

④ 任务集来自白名单 + 每步 `requires_grad` 守卫（见模型侧），本文件只做几何。

⑤ 自适应权重**只调制因子**，保持原权重总质量 `Σw` 不变：

       m_i = exp(−λ·c̄_i) / ( Σ_j w_j·exp(−λ·c̄_j) / Σ_j w_j )

   全 `c̄` 相等时 `m_i ≡ 1`（退化为原权重）。若按 GPT 原公式把 `w_i` 本身
   归一到和为 1，则权重和从 2.2 变成 1 ⇒ 所有梯度整体缩放 = 顺带改了有效
   LR，"无冲突时退化为原权重"这条不变量也不成立。

── 与 PCGrad 原文的一处有意偏离 ──

PCGrad 对每个任务 i 以**随机顺序**遍历其他任务 j。本实现用固定的升序
（j = 0..T−1, j≠i），换取可复现性与可单测性（随机顺序在有 `>2` 个任务时会
让结果依赖 RNG 状态，而本项目的所有数字都要求同口径可比）。T=2 时两者等价。
"""

from collections import OrderedDict

import torch


# ══════════════════════════════════════════════════════════════
#  参数切分
# ══════════════════════════════════════════════════════════════


def split_params(named_parameters, shared_prefixes):
    """把**可训**参数切成 `(shared, private)` 两个 `[(name, param), ...]`。

    shared  = 名字以 `shared_prefixes` 之一开头的参数（融合路径，被 ≥2 个任务
              触达）→ 做冲突检测 + 投影
    private = 其余（只被 1 个任务触达的 head / scorer）→ 直接求和写回

    冻结参数（`requires_grad=False`）两边都不进 —— 标准 backward 下它们本来就
    拿不到梯度，写进 `p.grad` 反而是引入新行为。
    """
    prefixes = tuple(shared_prefixes)
    shared, private = [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        if prefixes and name.startswith(prefixes):
            shared.append((name, p))
        else:
            private.append((name, p))
    return shared, private


# ══════════════════════════════════════════════════════════════
#  扁平化 / 反扁平化
# ══════════════════════════════════════════════════════════════


def flatten(grads, params, dtype=torch.float32):
    """把一组梯度张量拼成一个 1-D 向量；`None` 记为 0。

    `materialize_grads` 是 torch ≥ 2.1 才有的 API，目标环境是 torch 1.9，
    所以这里手写 `None → zeros`。
    """
    if not params:
        return torch.empty(0, dtype=dtype)
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros(p.numel(), dtype=dtype, device=p.device))
        else:
            parts.append(g.reshape(-1).to(dtype))
    return torch.cat(parts)


def unflatten(flat, params):
    """1-D 向量 → 与 `params` 同形状的梯度张量列表。"""
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(flat[off:off + n].view_as(p))
        off += n
    return out


# ══════════════════════════════════════════════════════════════
#  冲突统计
# ══════════════════════════════════════════════════════════════


def offdiag_mask(T, device=None):
    """`[T, T]` 的非对角 bool 掩码。

    故意用 `eye(...) == 0` 而不是 `~torch.eye(T, dtype=torch.bool)`：后者依赖
    `torch.eye` 对 bool dtype 的支持，而本文件要同时跑在 torch 1.9（autodl）
    与 ≥2.6（本地验证）上。
    """
    e = torch.eye(T, dtype=torch.float64, device=device)
    return e == 0


def pairwise_cosines(grads, eps=1e-12):
    """`[T, D]` 扁平梯度 → `[T, T]` pairwise cosine 矩阵（对角为 1，零向量为 0）。

    全程 float64：`D` 可以很大（本项目 SHARED 子空间约 2e4 维），float32 的
    `a·b` 在 `‖a‖‖b‖` 巨大时会吃掉有效位。
    """
    if len(grads) == 0:
        return torch.zeros(0, 0, dtype=torch.float64)
    G = torch.stack([g.to(torch.float64) for g in grads])          # [T, D]
    norms = G.norm(dim=1, keepdim=True)                            # [T, 1]
    Gn = G / norms.clamp(min=eps)
    C = Gn @ Gn.t()                                                # [T, T]
    # 任一为零向量的 pair（含对角）置 0，避免 0/0 的虚假"正交"
    alive = (norms > eps).squeeze(-1)                              # [T]
    alive_pair = alive.unsqueeze(0) & alive.unsqueeze(1)           # [T, T]
    return torch.where(alive_pair, C, torch.zeros_like(C))


def conflict_degrees(cosines):
    """`[T, T]` cosine → 长度 T 的 float64 冲突度向量。

    `c_i` = 任务 i 的**冲突度** = 与其他任务中 cos < 0 的比例，∈ [0, 1]。
    用计数而非"负 cosine 的幅值"：任务的梯度尺度随 batch 波动几个数量级，
    让尺度进入 c 会让自适应权重变成梯度的函数而不是冲突的函数。
    """
    T = cosines.shape[0]
    if T <= 1:
        return torch.zeros(T, dtype=torch.float64, device=cosines.device)
    off = offdiag_mask(T, cosines.device)     # ★ device 必须跟着输入走
    neg = (cosines < 0) & off
    return neg.sum(dim=1).to(torch.float64) / (T - 1)


# ══════════════════════════════════════════════════════════════
#  自适应权重（§⑤）
# ══════════════════════════════════════════════════════════════


def adaptive_modulation(conflict, base_weights, lam, device=None):
    """冲突度 `c̄` → 权重**调制因子** `m`（保持总质量）。

        m_i = exp(−λ·c̄_i) / ( Σ_j w_j·exp(−λ·c̄_j) / Σ_j w_j )

    不变量：全 `c̄` 相等 ⇒ `m_i ≡ 1`；且 `Σ w_i·m_i ≡ Σ w_i`。

    `device`：`conflict` 传 Python list 时无法从它推 device ⇒ 由调用方显式给
    （通常是梯度张量的 device）。传张量时 `device=None` 保持原 device。
    """
    c = torch.as_tensor(conflict, dtype=torch.float64, device=device)
    # ★ `base_weights` 通常是 Python list ⇒ `as_tensor` 会落在 **CPU**，而 `c` 在
    # CUDA 上 ⇒ `w * raw` 直接设备不一致崩掉。必须显式跟着 c 的 device。
    w = torch.as_tensor(base_weights, dtype=torch.float64, device=c.device)
    if c.numel() != w.numel():
        raise ValueError(
            f"conflict ({c.numel()}) and base_weights ({w.numel()}) must match"
        )
    raw = torch.exp(-lam * c)
    denom = (w * raw).sum()
    if denom <= 0:
        return torch.ones_like(w)
    return raw * w.sum() / denom


def update_ema(ema, value, beta, initialized):
    """`c̄ ← β·c̄ + (1−β)·c`（首个观测直接取 `c`）。

    device 跟着 `ema` 走：`value` 若是 Python list 会落在 CPU，而 `ema` 在 CUDA
    上 ⇒ `beta * ema + ... * value` 设备不一致。
    """
    value = torch.as_tensor(value, dtype=torch.float64,
                            device=ema.device if ema is not None else None)
    if not initialized or ema is None:
        return value.clone()
    return beta * ema + (1.0 - beta) * value


# ══════════════════════════════════════════════════════════════
#  PCGrad 投影（只在 SHARED 子空间内、对**未加权**梯度做）
# ══════════════════════════════════════════════════════════════


def pcgrad_project(grads, weights, eps=1e-12, conflict_threshold=0.0):
    """PCGrad 投影 + 加权求和，返回单个 1-D 合成梯度。

    `grads`   : 长度 T 的 1-D 张量列表（**未加权**的 `grad L_t`，SHARED 子空间）
    `weights` : 长度 T 的最终权重（含自适应调制因子）
    `conflict_threshold` : ★ 2026-09-26 新增。只在
        `cos(g_i, g_j) < conflict_threshold` 时才投影。

        * **0.0（默认）= 原行为，逐位不变**（`dot < 0` ⟺ `cos < 0`，非零向量下等价）
        * `-0.05` 之类 ⇒ **只有"真冲突"才投影**，`cos ≈ -0.001` 这种噪声级冲突被忽略。

        为什么需要它：实测 `mean_cos` 接近 0 ⇒ 现在的 `dot < 0` 判据会**对噪声级
        的负相关做投影**，等于往梯度里注入随机扰动。设一个容忍带只处理真冲突。

    对每个 i，按 j 升序遍历 j≠i：
        if g_i·g_j < thr:  g_i ← g_i − (g_i·g_j / ‖g_j‖²) · g_j
        其中 thr = conflict_threshold · ‖g_i‖ · ‖g_j‖（把 cosine 阈值换算成 dot 阈值）
    `‖g_j‖² = 0`（该任务在 SHARED 上无梯度）时跳过 —— 与 `g_i·g_j = 0` 同效，
    但显式跳过可以避免 0/0。
    """
    T = len(grads)
    if T == 0:
        return torch.empty(0)
    if len(weights) != T:
        raise ValueError(f"grads ({T}) and weights ({len(weights)}) must match")

    # float64 做投影几何：投影系数是比值，float32 在 ‖g‖ 跨数量级时会失准
    G = [g.to(torch.float64) for g in grads]
    sq = [float((g * g).sum()) for g in G]

    proj = []
    for i in range(T):
        gi = G[i].clone()
        for j in range(T):
            if i == j or sq[j] <= eps:
                continue
            dot = float((gi * G[j]).sum())
            if conflict_threshold != 0.0:
                # cos 阈值 → dot 阈值。默认分支不额外算范数 ⇒ 零开销、逐位不变。
                thr = conflict_threshold * float(gi.norm()) * float(G[j].norm())
            else:
                thr = 0.0
            if dot < thr:
                gi = gi - (dot / sq[j]) * G[j]
        proj.append(gi)

    # ★ device 从梯度张量取（`weights` 常是 Python list ⇒ 默认落 CPU）
    w = torch.as_tensor(weights, dtype=torch.float64, device=proj[0].device)
    out = torch.zeros_like(proj[0])
    for i in range(T):
        out = out + w[i] * proj[i]
    return out


def weighted_sum(grads, weights):
    """无投影时的加权求和（PRIVATE 部分与"投影关闭"的等价路径）。"""
    if len(grads) == 0:
        return torch.empty(0)
    # ★ device 从梯度张量取（同上）
    w = torch.as_tensor(weights, dtype=torch.float64, device=grads[0].device)
    out = torch.zeros_like(grads[0], dtype=torch.float64)
    for i, g in enumerate(grads):
        out = out + w[i] * g.to(torch.float64)
    return out


# ══════════════════════════════════════════════════════════════
#  写回 p.grad
# ══════════════════════════════════════════════════════════════


def touched_counts(per_task_grads, n_params):
    """→ 长度 `n_params` 的 int 列表：每个参数被**多少个任务**触达。

    `per_task_grads`：长度 T 的列表，每项是长度 `n_params` 的 `grad | None` 列表
    （`autograd.grad(..., allow_unused=True)` 的原样输出）。

    用途有二：① 写回时跳过"无任何任务触达"的参数（保持与原路径逐位一致）；
    ② **自审计** —— 名单外却 `>=2` 个任务触达 = `acgo_shared_prefixes` 漏了；
    名单内却只有 `1` 个任务触达 = 该参数在稀释 cosine（§③）。
    """
    counts = [0] * n_params
    for grads in per_task_grads:
        for i, g in enumerate(grads):
            if g is not None:
                counts[i] += 1
    return counts


def write_grads(params, flat, seen=None):
    """把扁平梯度写回 `p.grad`，返回写回的参数个数。

    - `seen[i] is False` 的参数**跳过不写**：标准 backward 下它们的 `.grad` 是
      `None`。本 config 是 `weight_decay: 0`，对 Adam 而言 `None` 与 `0` 等价，
      但跳过能保证行为逐位不变，并且未来一旦打开 weight_decay，"写 0"会变成
      静默的 L2 更新 —— 那是语义变化，不该由这里偷偷引入。
    - 已存在的 `.grad` 用 `add_`（梯度累积被打开时不丢历史）。
    - **必须 clone**：`flat[a:b].view_as(p)` 是 `flat` 的视图，直接赋给 `p.grad`
      会让所有参数共享同一块 storage。
    """
    if flat is None or flat.numel() == 0:
        return 0
    off, n_written = 0, 0
    for i, (_, p) in enumerate(params):
        n = p.numel()
        if seen is not None and not seen[i]:
            off += n
            continue
        g = flat[off:off + n].view_as(p).to(p.dtype)
        off += n
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.add_(g)
        n_written += 1
    return n_written


# ══════════════════════════════════════════════════════════════
#  一步 ACGO
# ══════════════════════════════════════════════════════════════


class AcgoState:
    """跨 iteration 的 ACGO 状态（只有 EMA 冲突度与步计数）。"""

    def __init__(self):
        self.ema_conflict = None
        self.ema_initialized = False
        self.steps = 0


def acgo_combine(shared_grads, private_grads, nominal_weights, state,
                 projection=True, adaptive=True, lam=1.0, beta=0.9, eps=1e-12,
                 conflict_threshold=0.0):
    """给定各任务的 (SHARED 梯度, PRIVATE 梯度)，返回

        (g_shared_flat, g_private_flat, diagnostics)

    ★ 契约（§②，改代码前必读）：`shared_grads` / `private_grads` 是长度 T 的
    **扁平** 1-D 张量列表，且**已经是 `grad(w_t·L_t)`** —— 损失权重在
    `_compute_loss` 记录任务时就乘好了。因此本函数里用来乘梯度的是**调制因子
    `m_i`**（自适应关时全为 1），**不是** `nominal_weights`。

    `nominal_weights`（= 各任务的标称权重 `w_i^0`，如 offset 0.2 / embed 1.0 /
    score 1.0）**只**用于 §⑤ 归一化公式里的"总权重质量"，不参与乘梯度。
    把它再乘一遍梯度 = 权重平方，是最容易犯的错。

    PRIVATE 直接求和（构造上不可能冲突，§①）。
    """
    T = len(shared_grads)
    diag = OrderedDict()

    cos = pairwise_cosines(shared_grads, eps=eps)
    c = conflict_degrees(cos)

    diag["acgo_num_tasks"] = float(T)
    diag["acgo_conflict_ratio"] = float((c.sum() / T).item()) if T > 1 else 0.0
    if T > 1:
        off = offdiag_mask(T, cos.device)   # ★ 跟着余弦矩阵的 device
        diag["acgo_mean_cosine"] = float(cos[off].mean().item())
    else:
        diag["acgo_mean_cosine"] = 0.0
    for i in range(T):
        diag[f"acgo_conflict_task{i}"] = float(c[i].item())
        g = shared_grads[i]
        diag[f"acgo_gnorm_task{i}"] = (
            float(g.to(torch.float64).norm().item()) if g.numel() else 0.0
        )

    # ---- 自适应权重（§⑤）：算出的是**调制因子**，不是新权重 ----
    if adaptive and T > 1:
        state.ema_conflict = update_ema(
            state.ema_conflict, c, beta, state.ema_initialized
        )
        state.ema_initialized = True
        m = adaptive_modulation(state.ema_conflict, nominal_weights, lam,
                                device=cos.device)
        weights = [float(m[i]) for i in range(T)]
        for i in range(T):
            diag[f"acgo_mod_task{i}"] = float(m[i].item())
    else:
        weights = [1.0] * T
        for i in range(T):
            diag[f"acgo_mod_task{i}"] = 1.0

    # ---- SHARED：投影（§③ 只在 SHARED 内做）----
    if projection and T > 1:
        g_shared = pcgrad_project(shared_grads, weights, eps=eps,
                                  conflict_threshold=conflict_threshold)
    else:
        g_shared = weighted_sum(shared_grads, weights)

    # ---- PRIVATE：直接求和（§①）----
    g_private = weighted_sum(private_grads, weights) if private_grads else None

    state.steps += 1
    return g_shared, g_private, diag


# ══════════════════════════════════════════════════════════════
#  模块有效性实验的第二个对照：GradNorm（Chen et al., ICML 2018）
# ══════════════════════════════════════════════════════════════
#
# 用途：创新点 3（ACGO）的同类替代品。ACGO 改的是「冲突梯度怎么投影 +
# 按近期冲突自适应调制」；GradNorm 改的是「各任务的**梯度幅值**怎么平衡」，
# 不做投影、不看任务间的夹角。两者是同一问题（多任务梯度不平衡）的两种
# 代表解法，正好互为对照。
#
# 原文算法（Chen et al. 2018, Alg. 1）：
#   G_i(t) = ‖∇_W  w_i(t)·L_i(t)‖₂          （只在**共享**参数上取）
#   r_i(t) = (L_i(t)/L_i(0)) / mean_j(L_j(t)/L_j(0))
#   target_i = mean_j(G_j(t)) · r_i(t)^α
#   L_gn(t) = Σ_i |G_i(t) − target_i|
#   w ← w − lr·∇_w L_gn ，然后归一化使 Σ_i w_i = T
#
# 三处**有意简化**（写论文时必须如实说明，别当成严格复现）：
#   1. 原文的 w 用独立的 SGD 更新；这里用解析梯度
#      ∂G_i/∂w_i = G_i/w_i（因为 G_i 对 w_i 线性），一步梯度下降，
#      不做二阶近似、不加动量。
#   2. 原文每步重新归一化 w 到 Σw = T（T = 任务数）；这里照做。
#   3. 本实现把**更新后的 w 乘到本步已算出的任务梯度**上（差一步）。
#      这是常见做法：w 与梯度同源线性，误差是一步的滞后。
#
# 与 ACGO 的关键差别：**不做 PCGrad 投影**。所以这一臂同时改变了
# 「权重」和「有没有投影」两件事 —— 把它当作"另一种多任务优化方法"
# 整体对照，而不是"只换权重、其余不变"的受控消融。


class GradNormState:
    """跨 iteration 的 GradNorm 状态：任务权重 w、初始损失 L(0)、步数。"""

    def __init__(self):
        self.w = None          # list[float]，长度 T
        self.L0 = None         # list[float]，长度 T
        self.steps = 0


def gradnorm_combine(shared_grads, private_grads, task_losses, state,
                     alpha=0.15, lr=0.025, eps=1e-12):
    """按 GradNorm 规则更新任务权重并合成梯度。

    Args:
        shared_grads: 长度 T 的**扁平** 1-D 张量列表（已含标称权重的梯度）
        private_grads: 同上，只被单个任务触达的参数
        task_losses:  长度 T 的**未加权**损失值（float 或 0-d tensor）
        state:        GradNormState（跨步保存 w / L0）
        alpha:        原文 α = 0.15（越大越补偿"学得慢"的任务）
        lr:           原文 w 的学习率 0.025
    Returns:
        (g_shared_flat, g_private_flat, diag)
    """
    T = len(shared_grads)
    if T == 0:
        return None, None, OrderedDict()
    losses = [max(float(l.item() if torch.is_tensor(l) else l), 1e-8)
              for l in task_losses]

    if state.w is None:
        state.w = [1.0] * T
        state.L0 = list(losses)

    # G_i：共享子空间上的梯度范数（float64 求范数，避免 fp32 累计误差）
    G = [float(g.to(torch.float64).norm().item()) if g.numel() else 0.0
         for g in shared_grads]
    mean_G = sum(G) / T

    # r_i：相对训练速度（损失下降得慢 ⇒ r_i > 1 ⇒ target 更高 ⇒ 权重被推大）
    ratios = [losses[i] / state.L0[i] for i in range(T)]
    mean_r = sum(ratios) / T
    ratios = [r / mean_r if mean_r > eps else 1.0 for r in ratios]
    target = [mean_G * (r ** alpha) for r in ratios]

    # ---- 更新 w：解析梯度 ∂G_i/∂w_i = G_i / w_i ----
    new_w = []
    for i in range(T):
        wi = max(state.w[i], 1e-8)
        if G[i] > 0.0:
            grad_w = (1.0 if G[i] > target[i] else -1.0) * (G[i] / wi)
            wi = wi - lr * grad_w
        new_w.append(max(wi, 1e-3))
    s = sum(new_w)
    state.w = [w * T / s for w in new_w]    # 原文：Σw = T

    # ---- 用新权重合成（SHARED 与 PRIVATE 同一套 w）----
    g_shared = None
    for i, g in enumerate(shared_grads):
        term = g * state.w[i]
        g_shared = term if g_shared is None else g_shared + term
    g_private = None
    for i, g in enumerate(private_grads):
        term = g * state.w[i]
        g_private = term if g_private is None else g_private + term

    diag = OrderedDict()
    diag["gradnorm_mean_G"] = float(mean_G)
    diag["gradnorm_mean_r"] = float(mean_r)
    for i in range(T):
        diag[f"gradnorm_w{i}"] = float(state.w[i])
        diag[f"gradnorm_G{i}"] = float(G[i])
        diag[f"gradnorm_target{i}"] = float(target[i])

    state.steps += 1
    return g_shared, g_private, diag
