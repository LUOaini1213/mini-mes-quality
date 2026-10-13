"""Image-dataset access for the visual-inspection part.

Real images are NOT stored in this repository. By default the code reads VisA (Amazon, CC BY 4.0) from
D:\\data\\visa\\raw (override with MESQ_VISA_ROOT) and uses the official one-class split file split_csv/1cls.csv.
An MVTec-AD-style folder (category/train/good, category/test/<type>, category/ground_truth/<type>/*_mask.png) is
also supported through load_mvtec_style(), so MVTec AD can be dropped in once it has been obtained from MVTec.
See DATA_SOURCES.md.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

VISA_ROOT = Path(os.environ.get("MESQ_VISA_ROOT", r"D:\data\visa\raw"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    path: str          # dataset-relative path (what gets written to inspection_result.image_ref)
    split: str         # train | test
    label: int         # 0 normal, 1 anomaly
    mask: str | None   # dataset-relative mask path or None


def load_visa_split(category: str, root: Path = VISA_ROOT) -> list[Sample]:
    out = []
    with open(root / "split_csv" / "1cls.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["object"] != category:
                continue
            out.append(Sample(row["image"], row["split"], int(row["label"] == "anomaly"), row["mask"] or None))
    if not out:
        raise FileNotFoundError(f"no rows for {category!r} in {root / 'split_csv' / '1cls.csv'}")
    return out


def load_mvtec_style(category_dir: Path) -> list[Sample]:
    category_dir = Path(category_dir)
    out = []
    for p in sorted((category_dir / "train" / "good").glob("*.png")):
        out.append(Sample(str(p.relative_to(category_dir.parent)), "train", 0, None))
    for d in sorted((category_dir / "test").iterdir()):
        for p in sorted(d.glob("*.png")):
            if d.name == "good":
                out.append(Sample(str(p.relative_to(category_dir.parent)), "test", 0, None))
            else:
                m = category_dir / "ground_truth" / d.name / f"{p.stem}_mask.png"
                out.append(Sample(str(p.relative_to(category_dir.parent)), "test", 1,
                                  str(m.relative_to(category_dir.parent))))
    return out


def load_image(root: Path, rel: str, size: int) -> np.ndarray:
    """RGB image resized (bilinear) to size x size, ImageNet-normalised, CHW float32."""
    im = Image.open(root / rel).convert("RGB").resize((size, size), Image.BILINEAR)
    a = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return a.transpose(2, 0, 1)


def load_mask(root: Path, rel: str | None, size: int) -> np.ndarray:
    """Binary mask (any non-zero pixel = anomaly), nearest-neighbour resized; all zeros when rel is None."""
    if rel is None:
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(root / rel).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.uint8)
