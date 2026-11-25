"""
Multi-view orthographic projection for 3D point clouds.
Pure NumPy implementation — zero PyTorch index ops, rock-solid.
"""

import numpy as np
import torch

VIEW_DIRS = [
    (0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1),
]
NUM_VIEWS = 6

# ---------------------------------------------------------------------------
# View ablation (2026-10-05)
#
# `NUM_VIEWS` above is the full 6-view default and is still what every existing
# config uses -- nothing below changes behaviour unless a caller opts in.
#
# For the 1..5-view ablation the view ORDER matters, because with V < 6 you are
# choosing *which* views survive. A separate order is used rather than a slice of
# VIEW_DIRS, because VIEW_DIRS happens to put the z-projections LAST, so its
# prefix would not be the intended sequence.
#
# ★ As of 2026-10-08 the top view is LAST (V6 only) -- see the block below for
#   what that means and what must be disclosed with it. The 2026-10-05..10-07
#   order (top view FIRST, the conservative arrangement) is kept as
#   VIEW_ORDER_TOP_FIRST so the two can be swapped back with a one-line change.
# Order used by the 1..5-view ablation.
#
# ★ 2026-10-08 CHANGE (user decision): the TOP view now goes LAST.
#   V1..V5 = the 5 non-top directions (4 sides + bottom). ONLY V6, the full
#   six-view model, contains the top view.
#
#   Rationale as recorded by the user: "1-5视角都不包括顶视，只有我们的异构6视角
#   才包括顶视" -- i.e. so that the top view's contribution shows up as the final
#   step of the curve, V5 -> V6.
#
#   ⚠️ WHAT THIS DOES AND DOES NOT DO -- write this in the paper, do not leave it
#   to a reviewer:
#     * It does NOT change V6. The six-view model is the same model either way,
#       so this cannot make "6 views > 1 view" any truer than it already is.
#       It only lowers points V1..V5 by withholding the strongest view from them.
#     * V1 is therefore NOT the strongest single view. The strongest single view
#       is top-only, measured under the previous (top-first) order as 0.8839
#       (view_ablation_v1.yaml, run S1_view1_150ep). Report that number too, or
#       the curve reads as "we removed the best view from our baseline".
#     * The framing a reviewer will reach for: "the authors withhold the
#       strongest view from the single-view baseline so that the six-view model
#       looks useful." The only defence is to state the design explicitly:
#         `the 1..5-view prefix contains no top view; the top view is added only
#          in the final step, so that its contribution is isolated`
#       and to disclose the top-only number next to it.
#
#   History: the previous order put the top view FIRST, which is the conservative
#   arrangement (starting from the strongest single view makes the multi-view
#   claim harder, not easier). It is preserved below as VIEW_ORDER_TOP_FIRST so
#   the two can be swapped back with a one-line change and the old numbers stay
#   traceable.
#
#  NEVER TWO VIEWS OF THE SAME AXIS BACK TO BACK. `(d, +1)` and `(d, -1)` are the
#  SAME axis seen from opposite sides -- highly correlated. The order below
#  interleaves the two side axes before revisiting either.
VIEW_ORDER_ABLATION = [
    (0, 1),      # 1. side, along +x
    (1, 1),      # 2. side, along +y
    (0, -1),     # 3. side, along -x
    (1, -1),     # 4. side, along -y
    (2, -1),     # 5. bottom-up (project along z, from below)
    (2, 1),      # 6. top-down   <- only V6 has this
]

# The previous (conservative) arrangement: top view FIRST. Kept so the order can
# be reverted by pointing `select_views` at this list, and so the top-only number
# (V1 under this order) stays traceable.
VIEW_ORDER_TOP_FIRST = [
    (2, 1),      # 1. top-down (project along z, from above)
    (0, 1),      # 2. side, along +x
    (1, 1),      # 3. side, along +y
    (0, -1),     # 4. side, along -x
    (1, -1),     # 5. side, along -y
    (2, -1),     # 6. bottom-up (the last, least informative one)
]


def select_views(num_views, order=None):
    """The first `num_views` entries of an ordered view list.

    num_views >= 6 or None -> all six views (identical to the old behaviour).
    """
    order = VIEW_ORDER_ABLATION if order is None else order
    if num_views is None or num_views >= len(order):
        return list(order)
    num_views = int(num_views)
    if num_views < 1:
        raise ValueError("num_views must be >= 1, got %r" % (num_views,))
    return list(order[:num_views])


def multi_view_project(pos, rgb, resolution=224, radius=8.0, view_dirs=None):
    """
    Project 3D point cloud to orthographic views (NumPy CPU).

    view_dirs: list of (proj_dim, sign). None -> the full 6-view VIEW_DIRS,
               i.e. the historical behaviour.

    Returns:
        images:         [V, 3, R, R]  float32  (on pos.device)
        pixel_to_point: [V, R*R]      int64    point index per flat pixel
    """
    if view_dirs is None:
        view_dirs = VIEW_DIRS
    target_device = pos.device
    pos_np = pos.cpu().numpy()
    rgb_np = rgb.cpu().numpy()
    N = pos_np.shape[0]
    R = resolution
    R2 = R * R

    image_list = []
    p2p_list = []

    for v, (proj_dim, sign) in enumerate(view_dirs):
        img_axes = [d for d in range(3) if d != proj_dim]
        u_axis, v_axis = img_axes[0], img_axes[1]

        u = pos_np[:, u_axis]
        v = pos_np[:, v_axis]
        u_norm = (u / radius) * 0.5 + 0.5
        v_norm = (v / radius) * 0.5 + 0.5
        u_pix = np.clip((u_norm * (R - 1)).astype(np.int64), 0, R - 1)
        v_pix = np.clip((v_norm * (R - 1)).astype(np.int64), 0, R - 1)
        pixel_idx = v_pix * R + u_pix  # [N]

        # Vectorized: unbuffered accumulation over duplicate pixel indices
        img_flat = np.zeros((R2, 3), dtype=np.float32)
        count = np.zeros(R2, dtype=np.float32)
        np.add.at(img_flat, pixel_idx, rgb_np)
        np.add.at(count, pixel_idx, np.ones(N, dtype=np.float32))

        count = np.maximum(count, 1.0)
        img_flat = img_flat / count[:, None]
        image_list.append(img_flat.reshape(1, R, R, 3))

        # --- Pixel-to-point mapping (closest-to-camera, vectorized) ---
        p2p = np.full(R2, -1, dtype=np.int64)
        depth = pos_np[:, proj_dim] * sign
        # lexsort: primary key = pixel_idx, secondary key = -depth.
        # After sorting, the first occurrence of each pixel index
        # corresponds to the point with maximum depth (closest to camera).
        order = np.lexsort((-depth, pixel_idx))
        sorted_pixels = pixel_idx[order]
        unique_pixels, first_idx = np.unique(sorted_pixels, return_index=True)
        p2p[unique_pixels] = order[first_idx]
        p2p_list.append(p2p.reshape(1, R2))

    images = np.concatenate(image_list, axis=0)         # [6, R, R, 3]
    pixel_to_point = np.concatenate(p2p_list, axis=0)   # [6, R2]

    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).to(target_device)
    p2p_t = torch.from_numpy(pixel_to_point).to(target_device)
    return images_t, p2p_t


def pixel_to_patch_map(resolution=224, patch_size=32):
    """Flat-pixel to patch index mapping."""
    R = resolution
    P = patch_size
    grid_size = R // P
    y = np.arange(R).reshape(-1, 1).repeat(R, axis=1)
    x = np.arange(R).reshape(1, -1).repeat(R, axis=0)
    patch_y = y // P
    patch_x = x // P
    patch_idx = (patch_y * grid_size + patch_x).flatten()
    return torch.from_numpy(patch_idx).long()
