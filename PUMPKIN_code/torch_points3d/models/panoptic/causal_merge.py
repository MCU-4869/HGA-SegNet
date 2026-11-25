"""Causal instance merge — over-segment-then-merge for the growth-feature run.

Motivation
----------
The growth feature (terrain-normalised canopy height h = z - DTM) injected into
the Offset/Embed heads breaks the test-RECALL wall (0.840 -> 0.859) but pays for
it with a PRECISION drop (0.975 -> 0.963): point-level height over-separates one
tree into fragments. This module recovers precision without giving the recall
back, by learning which adjacent fragments are *causally* the same instance.

Why this is non-redundant (the old 2D->3D causal attention was not)
--------------------------------------------------------------------
- It consumes a signal the rest of the pipeline never sees for this decision:
  the horizontal (xy) crown-footprint overlap as a confounder-free moderator,
  plus the growth feature's vertical push (|h mean diff|) as the treatment.
  Clustering uses embedding/offset, the scorer uses fused features — neither
  reads `xy footprint` for merge-vs-split.
- It is trained with its OWN objective (binary same-tree supervision from GT
  instance labels), not backprop from the shared prediction loss, so the gradient
  has a dedicated channel (CausalNET lesson: separate objective + new input).

Counterfactual framing
----------------------
P(same | pair) answers "would these two fragments be one tree?" — a merge-vs-split
treatment. At inference we apply the discrete decision (union-find merge) so the
downstream scorer sees whole trees instead of fragments.
"""
import torch
import torch.nn as nn


def cluster_summaries(clusters, pos, height, batch):
    """Per-cluster summary features.

    Args:
        clusters: list of [n_i] long tensors of point indices.
        pos:      [N, 3] point positions (device).
        height:   [N, 1] terrain-normalised canopy height (device).
        batch:    [N]    batch-element id per point (device).

    Returns:
        centroids [K,3], h_mean [K], h_max [K], h_min [K],
        xy_min [K,2], xy_max [K,2], sizes [K] long, cbatch [K] long
    """
    device = pos.device
    K = len(clusters)
    centroids = torch.zeros(K, 3, device=device)
    h_mean = torch.zeros(K, device=device)
    h_max = torch.zeros(K, device=device)
    h_min = torch.zeros(K, device=device)
    xy_min = torch.zeros(K, 2, device=device)
    xy_max = torch.zeros(K, 2, device=device)
    sizes = torch.zeros(K, dtype=torch.long, device=device)
    cbatch = torch.zeros(K, dtype=torch.long, device=device)
    for i, c in enumerate(clusters):
        p = pos[c]
        h = height[c].view(-1)
        centroids[i] = p.mean(0)
        h_mean[i] = h.mean()
        h_max[i] = h.max()
        h_min[i] = h.min()
        xy_min[i] = p[:, :2].min(0).values
        xy_max[i] = p[:, :2].max(0).values
        sizes[i] = c.numel()
        cbatch[i] = batch[c[0]]
    return centroids, h_mean, h_max, h_min, xy_min, xy_max, sizes, cbatch


def build_merge_pairs(centroids, cbatch, ctype, max_dist=1e6):
    """Candidate fragment pairs: same batch, same cluster type, within max_dist.

    Returns list of (i, j) index pairs with i < j.
    """
    K = centroids.shape[0]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            if cbatch[i] != cbatch[j]:
                continue
            if ctype is not None and ctype[i] != ctype[j]:
                continue
            if torch.norm(centroids[i] - centroids[j]) > max_dist:
                continue
            pairs.append((i, j))
    return pairs


def pair_feature_tensor(centroids, h_mean, h_max, h_min, xy_min, xy_max, sizes, pairs):
    """Feature vector per pair, lightly normalised for the small MLP.

    Causal-moderation framing: the growth feature splits one tree into
    vertically stacked fragments (same tree, different terrain-normalised
    height). Whether that split is *spurious* (merge back) or *useful*
    (keep two adjacent trees separate) is decided by a confounder-free
    moderator: the horizontal (xy) crown-footprint overlap.

    - treatment magnitude : |h_mean_i - h_mean_j|  (growth's vertical push)
    - moderator           : xy_iou (horizontal footprint overlap)
    - stacking            : hov  (vertical z-overlap, negative = stacked)

    The old features |h_max diff| and |h_min diff| are dropped: they penalise
    the height difference that *defines* a same-tree vertical stack, i.e. their
    sign was backwards for exactly the pairs growth wrongly splits.
    """
    feats = []
    for (i, j) in pairs:
        d = torch.norm(centroids[i] - centroids[j])
        dxy = torch.norm((centroids[i] - centroids[j])[:2])
        hov = torch.min(h_max[i], h_max[j]) - torch.max(h_min[i], h_min[j])
        # Horizontal footprint overlap (IoU of xy bounding boxes), 0..1.
        xa, ya = xy_min[i][0], xy_min[i][1]
        xb, yb = xy_max[i][0], xy_max[i][1]
        xa2, ya2 = xy_min[j][0], xy_min[j][1]
        xb2, yb2 = xy_max[j][0], xy_max[j][1]
        ix = torch.clamp(torch.min(xb, xb2) - torch.max(xa, xa2), min=0.0)
        iy = torch.clamp(torch.min(yb, yb2) - torch.max(ya, ya2), min=0.0)
        inter = ix * iy
        area_i = (xb - xa) * (yb - ya)
        area_j = (xb2 - xa2) * (yb2 - ya2)
        xy_iou = inter / (area_i + area_j - inter + 1e-6)
        f = torch.stack([
            d / 10.0,
            dxy / 10.0,
            xy_iou,
            (h_mean[i] - h_mean[j]).abs() / 50.0,
            hov / 50.0,
            torch.log1p(sizes[i].float()) / 10.0,
            torch.log1p(sizes[j].float()) / 10.0,
            (torch.log1p(sizes[i].float()) - torch.log1p(sizes[j].float())).abs(),
        ])
        feats.append(f)
    if not feats:
        return torch.zeros(0, 8, device=centroids.device)
    return torch.stack(feats, dim=0)


def cluster_majority_labels(clusters, instance_labels):
    """GT majority instance label per cluster (computed once, not per pair)."""
    out = torch.zeros(len(clusters), device=instance_labels.device,
                      dtype=instance_labels.dtype)
    for i, c in enumerate(clusters):
        out[i] = _majority(instance_labels[c])
    return out


def pair_labels_from_gt(majority, pairs):
    """Binary same-tree label for each pair, from precomputed GT majority labels."""
    labels = torch.zeros(len(pairs), device=majority.device)
    for k, (i, j) in enumerate(pairs):
        labels[k] = 1.0 if majority[i] == majority[j] else 0.0
    return labels


def _majority(x):
    uniq, counts = torch.unique(x, return_counts=True)
    return uniq[counts.argmax()]


def union_find_merge(clusters, ctype, pairs, merge_mask):
    """Merge clusters via union-find over the pairs flagged by merge_mask.

    Args:
        clusters: list of [n_i] tensors.
        ctype:    [K] cluster-type tensor (may be None).
        pairs:    list of (i, j).
        merge_mask: [P] bool — True = merge this pair.

    Returns (merged_clusters, merged_ctype or None).
    """
    K = len(clusters)
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for k, (i, j) in enumerate(pairs):
        if merge_mask[k]:
            union(i, j)

    root_to_idx = {}
    merged = []
    merged_type = []
    for i in range(K):
        r = find(i)
        if r not in root_to_idx:
            root_to_idx[r] = len(merged)
            merged.append([])
            merged_type.append(ctype[i].item() if ctype is not None else 0)
        merged[root_to_idx[r]].append(clusters[i])

    out_clusters = []
    for members in merged:
        if len(members) == 1:
            out_clusters.append(members[0])
        else:
            out_clusters.append(torch.cat(members))
    if ctype is not None:
        merged_type = torch.tensor(merged_type, device=ctype.device, dtype=ctype.dtype)
    else:
        merged_type = None
    return out_clusters, merged_type


class CausalMerge(nn.Module):
    """Small same-tree classifier over fragment pairs."""

    def __init__(self, feat_dim=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        # feat: [P, feat_dim] -> [P] logits
        return self.net(feat).squeeze(-1)
