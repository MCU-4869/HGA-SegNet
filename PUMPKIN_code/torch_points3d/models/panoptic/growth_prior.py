"""生长先验的取数（模块有效性实验 · 创新点 2 的对照开关）。

单独放一个文件是为了能**脱离 MinkowskiEngine 单测**（见 test_mech_modules.py 第 4 节）。
`PointGroup3heads_Hetero` 只做 `from .growth_prior import resolve_growth_prior`。

两种先验（`growth_prior_type`）：

    dtm   —— batch 的 `height`：数据集 process() 里算的 h = z − DTM（地形归一化冠层高度）。
             注意它依赖**真值地面标签**建 DTM（见 memory: growth-feature-uses-gt-labels）。
    raw_z —— batch 的 `pos_z`：`XYZFeature(add_z=True)` 在 `Center`
             **之前**捕获的绝对高程（cm）。这条臂用来回答"能不能不要 DTM"。

⛔ 为什么不能用 `raw_pos[:, 2]`
    `raw_pos = data.pos` 已经过 PyG 的 `Center`（逐样本减去本样本质心），
    所以它是「样本内居中高度」，与绝对高程差一个**逐样本常数**。
    2026-10-24 的第一版实现就是取它，于是 config / 日志 / checkpoint 里写着
    "绝对高程 z"，实际喂进去的却是居中高度 —— 两个对照的结论完全不同
    （居中高度已经部分去掉地形起伏，几乎是个免费的地形归一化）。
    ⇒ 拿不到 `pos_z` 就 **raise**（fail fast）。静默降级要跑满 44 h 才会暴露。

其它已知不对称（论文里要如实写）
    `pos_z` 在 `RandomScaleAnisotropic`（0.9–1.1 随机缩放）**之后**捕获，因此带缩放；
    而缓存里的 `height` 是 process() 时算的，不带缩放。两者都逐样本、都与几何同步，
    量级同源，但"带不带随机缩放"这点不完全对齐。
"""


def resolve_growth_prior(input_data, raw_pos, prior_type):
    """→ (height[N,1] 或 None, source 字符串)。

    Args:
        input_data: collate 之后的 batch（有 `.height` / `.pos_z` 属性）
        raw_pos:    [N,3]，**保留参数只为报错信息与将来排查**；raw_z 分支**不使用**它
        prior_type: 'dtm' | 'raw_z'

    Returns:
        (height, source)：dtm 分支下 height 可能是 None（旧缓存没有 height 属性，
        调用方据此打 `★MISSING★`）；source 写进日志，供起跑核对 grep。
    """
    prior_type = str(prior_type).lower()

    if prior_type == "raw_z":
        height = getattr(input_data, "pos_z", None)
        if height is None:
            raise RuntimeError(
                "[GrowthFeature] growth_prior_type=raw_z 需要 batch 里的 `pos_z`"
                "（= XYZFeature(add_z: True) 在 Center 之前捕获的绝对高程），但没找到。"
                "排查：① data 配置的 train_transforms / test_transform 里是否还有 "
                "`XYZFeature(add_z: True)`；② 它是否被挪到了 `Center` 之后"
                "（那样 pos_z 就变成居中高度）；③ 是否有变换把它删掉了。"
                "★ 不要退回 raw_pos[:,2] —— 那是逐样本居中高度，不是绝对高程。")
        return height.view(-1, 1), "raw_z_abs(pos_z, pre-Center)"

    if prior_type != "dtm":
        raise ValueError(
            f"[GrowthFeature] growth_prior_type 只能是 'dtm' 或 'raw_z'，收到 {prior_type!r}")

    return getattr(input_data, "height", None), "dtm(h=z-DTM)"
