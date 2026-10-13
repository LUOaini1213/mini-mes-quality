"""Run the visual-inspection benchmark on the official VisA one-class test split.

Writes (no images, only numbers):
  results/vision/metrics.json             per category x method: image AUROC, pixel AUROC, AUPRO@0.3, threshold stats
  results/vision/<category>_scores.csv    per test image: dataset-relative path, label, scores, PatchCore decision
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import VISA_ROOT, load_image, load_mask, load_visa_split
from .metrics import aupro, auroc, pixel_auroc, rates_at_threshold
from .models import PatchMemory, PixelZScore, WRN50Features, device, upsample_and_smooth


@dataclass
class VisionConfig:
    categories: list = field(default_factory=lambda: ["pcb1", "pcb4"])
    image_size: int = 256          # resize (no centre crop, so border defects stay in view)
    bank_images: int = 400         # normal training images in the memory bank (subsampled for a 4 GB GPU / 16 GB RAM)
    val_images: int = 100          # held-out normal training images used ONLY to set the decision threshold
    coreset_ratio: float = 0.10
    threshold_quantile: float = 0.95   # threshold = this quantile of held-out normal image scores (fixed a priori)
    knn_k: int = 5
    batch_size: int = 4
    gpu_mem_fraction: float = 0.35   # the GPU is shared; cap this process (OOM instead of spilling to system RAM)
    seed: int = 0


def _batches(root, rels, size, bs):
    for i in range(0, len(rels), bs):
        yield np.stack([load_image(root, r, size) for r in rels[i:i + bs]])


def run_category(cat: str, cfg: VisionConfig, root: Path, dev) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    samples = load_visa_split(cat, root)
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(train))
    val = [train[i] for i in perm[:cfg.val_images]]
    bank = [train[i] for i in perm[cfg.val_images:cfg.val_images + cfg.bank_images]]

    net = WRN50Features().to(dev)
    S = cfg.image_size

    def extract(rels, keep_global=False, keep_pixels=False):
        feats, globs, pix = [], [], []
        for x in _batches(root, rels, S, cfg.batch_size):
            xt = torch.from_numpy(x).to(dev)
            feats.append(net(xt).half().cpu())
            if keep_global:
                globs.append(net.global_embedding(xt).float().cpu())
            if keep_pixels:
                pix.append(x)
        return (torch.cat(feats), torch.cat(globs) if keep_global else None,
                np.concatenate(pix) if keep_pixels else None)

    def log(msg):
        print(f"[vision] {cat}: {msg} ({time.time() - t0:.0f}s)", flush=True)

    fb, gb, pb = extract([s.path for s in bank], keep_global=True, keep_pixels=True)
    log(f"features of {len(bank)} memory-bank images")
    h, w, d = fb.shape[1:]
    mem = PatchMemory(cfg.coreset_ratio, cfg.seed, dev).fit(fb.reshape(-1, d))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    log(f"coreset {mem.bank.shape[0]} patches")
    n_patches = fb.shape[0] * h * w
    del fb
    zs = PixelZScore().fit(pb)
    del pb

    def patchcore_scores(rels):
        fs, g, px = extract(rels, keep_global=True, keep_pixels=True)
        maps = torch.cat([mem.score_maps(fs[i:i + cfg.batch_size]) for i in range(0, len(fs), cfg.batch_size)])
        return maps, g, px

    vmaps, vglob, vpix = patchcore_scores([s.path for s in val])
    tmaps, tglob, tpix = patchcore_scores([s.path for s in test])
    log("scored validation and test images")
    gb_d = gb.to(dev)

    def knn(g):
        dist = torch.cdist(g.to(dev), gb_d)
        return dist.topk(cfg.knn_k, largest=False).values.mean(1).cpu().numpy()

    labels = np.array([s.label for s in test])
    masks = np.stack([load_mask(root, s.mask, S) for s in test])
    pc_img = tmaps.flatten(1).max(1).values.cpu().numpy()
    pc_val = vmaps.flatten(1).max(1).values.cpu().numpy()
    pc_maps = upsample_and_smooth(tmaps, S)
    zs_maps_small = zs.score_maps(tpix)
    zs_maps = upsample_and_smooth(torch.from_numpy(zs_maps_small), S, sigma=2.0)
    zs_img = zs_maps_small.reshape(len(test), -1).max(1)
    zs_val = zs.score_maps(vpix).reshape(len(val), -1).max(1)
    knn_img, knn_val = knn(tglob), knn(vglob)

    out = {}
    for name, img, valsc, maps in [("patchcore_lite", pc_img, pc_val, pc_maps), ("global_knn", knn_img, knn_val, None),
                                   ("pixel_zscore", zs_img, zs_val, zs_maps)]:
        thr = float(np.quantile(valsc, cfg.threshold_quantile))
        out[name] = dict(image_auroc=auroc(labels, img),
                         pixel_auroc=pixel_auroc(masks, maps) if maps is not None else None,
                         aupro_0_3=aupro(masks, maps) if maps is not None else None,
                         at_threshold=rates_at_threshold(labels, img, thr),
                         val_normal_fpr=float((valsc > thr).mean()))
    meta = dict(category=cat, n_train_available=len(train), n_bank_images=len(bank), n_val_images=len(val),
                n_test=len(test), n_test_anomalous=int(labels.sum()), memory_bank_patches=int(mem.bank.shape[0]),
                candidate_patches=int(n_patches), feature_map=[int(h), int(w)], feature_dim=int(d),
                seconds=round(time.time() - t0, 1), device=str(dev))
    thr_pc = out["patchcore_lite"]["at_threshold"]["threshold"]
    df = pd.DataFrame(dict(image=[s.path for s in test], label=labels, patchcore_score=pc_img, global_knn_score=knn_img,
                           pixel_zscore_score=zs_img, patchcore_decision=np.where(pc_img > thr_pc, "FAIL", "PASS")))
    del net, mem, gb_d
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"meta": meta, "methods": out}, df


def run(out_dir: Path, cfg: VisionConfig | None = None, root: Path = VISA_ROOT) -> dict:
    cfg = cfg or VisionConfig()
    torch.manual_seed(cfg.seed)
    dev = device()
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_mem_fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"config": asdict(cfg), "dataset": "VisA (1cls split)", "categories": {}}
    for cat in cfg.categories:
        r, df = run_category(cat, cfg, root, dev)
        res["categories"][cat] = r
        df.to_csv(out_dir / f"{cat}_scores.csv", index=False, float_format="%.6f")
        print(f"[vision] {cat}: " + ", ".join(f"{m} img-AUROC {v['image_auroc']:.3f}" for m, v in r["methods"].items()),
              flush=True)
    for m in ["patchcore_lite", "global_knn", "pixel_zscore"]:
        vals = [res["categories"][c]["methods"][m] for c in cfg.categories]
        res.setdefault("mean", {})[m] = {
            k: (float(np.mean([v[k] for v in vals])) if vals[0][k] is not None else None)
            for k in ["image_auroc", "pixel_auroc", "aupro_0_3"]}
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
