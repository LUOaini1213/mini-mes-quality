"""Memory bank / coreset / scoring logic on small synthetic arrays (no weights, no images)."""
import numpy as np
import torch

from mesq.vision.models import PatchMemory, PixelZScore, greedy_coreset, nn_distance, upsample_and_smooth


def test_nn_distance_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    q, b = torch.randn(300, 8, generator=g), torch.randn(50, 8, generator=g)
    ref = torch.stack([(b - x).norm(dim=1).min() for x in q])
    assert torch.allclose(nn_distance(q, b, chunk=64), ref, atol=1e-4)


def test_coreset_is_distinct_and_covers_clusters():
    g = torch.Generator().manual_seed(1)
    centres = torch.tensor([[0.0, 0], [10, 0], [0, 10], [10, 10]])
    pts = torch.cat([c + 0.1 * torch.randn(100, 2, generator=g) for c in centres])
    idx = greedy_coreset(pts, 4, proj_dim=0)
    assert len(set(idx.tolist())) == 4
    chosen = pts[idx]
    # k-center on 4 well separated clusters must pick one point from each cluster
    assert sorted(torch.cdist(chosen, centres).argmin(1).tolist()) == [0, 1, 2, 3]
    assert greedy_coreset(pts, 1000).shape[0] == pts.shape[0]


def test_patch_memory_scores():
    g = torch.Generator().manual_seed(2)
    normal = torch.randn(2000, 16, generator=g)
    mem = PatchMemory(coreset_ratio=0.5, seed=0).fit(normal)
    assert mem.bank.shape[0] == 1000
    inliers = normal[:4].reshape(1, 2, 2, 16)
    outliers = (normal[:4] + 25.0).reshape(1, 2, 2, 16)
    s_in, s_out = mem.score_maps(inliers), mem.score_maps(outliers)
    assert s_in.shape == (1, 2, 2)
    assert s_out.min() > s_in.max()
    bank_pts = mem.bank[:4].reshape(1, 2, 2, 16)
    assert torch.allclose(mem.score_maps(bank_pts), torch.zeros(1, 2, 2), atol=1e-2)  # cdist matmul path rounding


def test_upsample_and_smooth_shape_and_peak():
    m = torch.zeros(1, 8, 8)
    m[0, 2, 2] = 1.0
    up = upsample_and_smooth(m, 64, sigma=2)
    assert up.shape == (1, 64, 64)
    r, c = np.unravel_index(up[0].argmax(), up[0].shape)
    assert 12 <= r <= 28 and 12 <= c <= 28


def test_pixel_zscore_flags_changed_region():
    rng = np.random.default_rng(0)
    imgs = rng.normal(0, 0.1, size=(20, 3, 32, 32)).astype(np.float32)
    zs = PixelZScore(size=16).fit(imgs)
    test = imgs[:2].copy()
    test[1, :, 0:8, 0:8] += 3.0
    maps = zs.score_maps(test)
    assert maps.shape == (2, 16, 16)
    assert maps[1, 0:4, 0:4].mean() > 5 * maps[0].mean()
