"""Reproduce every result in results/ with one command:  python run_all.py

Steps
1. Visual inspection benchmark on VisA pcb1 + pcb4 (needs the dataset under D:\\data\\visa\\raw or MESQ_VISA_ROOT;
   see DATA_SOURCES.md). Skipped with a message if the images are not present, in which case the MES part falls
   back to a simulated detector.
2. Synthetic MES generation + traceability analysis (primary seed in detail + 20 seeds summary).
3. SOP assistant evaluation on the frozen question set.
4. README results block and results/report.html are regenerated from the JSON files.

Options: --reuse-vision  reuse results/vision/*.csv instead of recomputing (saves ~15 min on a GTX 1650)
         --seeds N       number of generator seeds for the multi-seed summary (default 20)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# resource etiquette: the machine is shared - cap CPU threads before numpy / torch are imported
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BUILD = ROOT / "build"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-vision", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    torch.set_num_threads(4)

    from mesq import experiment, report, sop_eval
    from mesq.vision.data import VISA_ROOT

    t0 = time.time()
    vis_dir = RESULTS / "vision"
    pool = None
    have_images = (VISA_ROOT / "split_csv" / "1cls.csv").exists()
    if args.reuse_vision and (vis_dir / "metrics.json").exists():
        print("[run_all] reusing results/vision", flush=True)
    elif have_images:
        from mesq.vision import run as vision_run
        vision_run.run(vis_dir)
    else:
        print(f"[run_all] VisA not found under {VISA_ROOT}; vision step skipped (simulated detector used)", flush=True)
    csvs = sorted(vis_dir.glob("*_scores.csv"))
    if csvs:
        from mesq.link import image_pool
        pool = image_pool([pd.read_csv(p) for p in csvs])

    print("[run_all] MES traceability ...", flush=True)
    experiment.run(RESULTS / "mes", pool, seeds=range(1, args.seeds + 1), build_dir=BUILD)
    print("[run_all] SOP assistant ...", flush=True)
    sop_eval.run(RESULTS / "sop")
    report.write_all(RESULTS, ROOT / "README.md")
    (RESULTS / "run_info.json").write_text(json.dumps(dict(
        python=sys.version.split()[0], torch=torch.__version__, cuda=torch.cuda.is_available(),
        seconds=round(time.time() - t0, 1), seeds=args.seeds, reused_vision=bool(args.reuse_vision)), indent=2),
        encoding="utf-8")
    print(f"[run_all] done in {time.time() - t0:.0f}s -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
