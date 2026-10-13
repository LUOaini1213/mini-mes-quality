"""Unsupervised anomaly detectors.

* PatchCoreLite: ImageNet-pretrained WideResNet-50-2 (torchvision weights), mid-level features (layer2 + layer3)
  with 3x3 local averaging, channel-reduced to 1024-d, a greedy k-center coreset memory bank of normal patches, and
  nearest-neighbour patch distance as the anomaly score (Roth et al., "Towards Total Recall in Industrial Anomaly
  Detection", CVPR 2022). "Lite": no re-weighting of the image score, a subsampled memory bank (see config) and a
  single resolution, to fit a 4 GB GPU.
* GlobalKNN: the same backbone, global-average-pooled layer2+layer3 embedding per image, kNN distance to the training
  embeddings (image score only; no localisation).
* PixelZScore: no learning at all - per-pixel mean / std of the normal training images (downsampled), anomaly map =
  max over channels of |z|. A deliberately naive reference that ignores misalignment.

Feature extraction is separated from the memory-bank logic so the bank/coreset/scoring code can be unit-tested on
small synthetic arrays without downloading weights.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WRN50Features(torch.nn.Module):
    """layer2 + layer3 of torchvision wide_resnet50_2 (IMAGENET1K_V1 weights, as used by PatchCore)."""

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.out_dim = out_dim
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)                                   # B x 1536 x h x w
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, 1, c)
        f = F.adaptive_avg_pool1d(f, self.out_dim).reshape(b, h, w, self.out_dim)
        return f                                                     # B x h x w x D

    @torch.no_grad()
    def global_embedding(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        return torch.cat([f2.mean((2, 3)), f3.mean((2, 3))], 1)


def project(feats: torch.Tensor, proj_dim: int, seed: int, dev: torch.device, chunk: int = 16384) -> torch.Tensor:
    """Random Gaussian projection (Johnson-Lindenstrauss) computed chunk-wise on `dev`."""
    if not proj_dim or feats.shape[1] <= proj_dim:
        return feats.to(dev, torch.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = (torch.randn(feats.shape[1], proj_dim, generator=g) / proj_dim ** 0.5).to(dev)
    return torch.cat([feats[i:i + chunk].to(dev, torch.float32) @ proj for i in range(0, feats.shape[0], chunk)])


def greedy_coreset(feats: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0,
                   dev: torch.device | None = None) -> torch.Tensor:
    """Greedy k-center selection (approximate, on a random projection) - returns selected row indices."""
    n = feats.shape[0]
    if n_select >= n:
        return torch.arange(n)
    dev = dev or feats.device
    z = project(feats, proj_dim, seed, dev)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    start = int(torch.randint(n, (1,), generator=g))
    sel = [start]
    zn = (z * z).sum(1)                       # squared norms; distances via |a|^2 + |b|^2 - 2ab (no n x d temporaries)

    def sqdist(i):
        return (zn + zn[i] - 2.0 * (z @ z[i])).clamp_min_(0)

    mind = sqdist(start)
    for _ in range(n_select - 1):
        i = int(torch.argmax(mind))
        sel.append(i)
        mind = torch.minimum(mind, sqdist(i))
    return torch.tensor(sel)


def nn_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Euclidean distance from each query row to its nearest bank row."""
    out = []
    for i in range(0, queries.shape[0], chunk):
        out.append(torch.cdist(queries[i:i + chunk], bank).min(1).values)
    return torch.cat(out)


class PatchMemory:
    """Memory bank of normal patch features + nearest-neighbour scoring (backbone-agnostic)."""

    def __init__(self, coreset_ratio: float = 0.1, seed: int = 0, dev: torch.device | None = None):
        self.ratio, self.seed = coreset_ratio, seed
        self.dev = dev or torch.device("cpu")
        self.bank: torch.Tensor | None = None

    def fit(self, patch_feats: torch.Tensor):
        """patch_feats: (N_patches, D) tensor of normal training patches (may live on the CPU, any float dtype)."""
        n_sel = max(1, int(round(self.ratio * patch_feats.shape[0])))
        idx = greedy_coreset(patch_feats, n_sel, seed=self.seed, dev=self.dev)
        self.bank = patch_feats[idx.cpu()].to(self.dev, torch.float32).contiguous()
        return self

    def score_maps(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, h, w, D) -> (B, h, w) patch anomaly scores."""
        b, h, w, d = feats.shape
        return nn_distance(feats.reshape(-1, d).to(self.dev, torch.float32), self.bank).reshape(b, h, w)


def upsample_and_smooth(maps: torch.Tensor, size: int, sigma: float = 4.0) -> np.ndarray:
    """Bilinear upsampling to the image size followed by Gaussian smoothing (as in PatchCore)."""
    from scipy.ndimage import gaussian_filter
    up = F.interpolate(maps[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    return np.stack([gaussian_filter(m, sigma) for m in up]).astype(np.float32)


class PixelZScore:
    def __init__(self, size: int = 64):
        self.size = size

    def _down(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x)
        return F.interpolate(t, size=(self.size, self.size), mode="area").numpy()

    def fit(self, imgs: np.ndarray):
        d = self._down(imgs)
        self.mu, self.sd = d.mean(0), d.std(0) + 1e-3
        return self

    def score_maps(self, imgs: np.ndarray) -> np.ndarray:
        z = np.abs((self._down(imgs) - self.mu) / self.sd).max(1)
        return z.astype(np.float32)
