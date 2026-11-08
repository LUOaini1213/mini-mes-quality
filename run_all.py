"""Reproduce every result in results/ with one command:  python run_all.py

Steps
1. Visual inspection benchmark on VisA pcb1 + pcb4 (needs the dataset under data/visa/raw in this repository, or
   set MESQ_VISA_ROOT; see DATA_SOURCES.md). Skipped with a message if the images are not present; the MES part
   then uses the committed per-image scores in results/vision/*_scores.csv (or a simulated detector if there are
   none).
2. Synthetic MES generation + traceability analysis (primary seed in detail + 20 seeds summary).
3. SOP assistant evaluation on the frozen question set. The hybrid system needs BAAI/bge-small-en-v1.5 in the local
   Hugging Face cache (see README "Reproduce"). If the committed results contain a hybrid row and the model cannot
   be loaded, run_all.py stops before writing anything, so the README never silently loses that row.
4. README results block and results/report.html are regenerated from the JSON files.

Options: --reuse-vision  reuse results/vision/*.csv instead of recomputing (saves about 20 min on a GTX 1650:
                         391 s + 805 s according to results/vision/metrics.json)
         --bm25-only     run the SOP evaluation without the embedding model and accept that the hybrid row is
                         dropped from results/ and the README
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


def check_sop_systems(results: Path = RESULTS, allow_drop: bool = False) -> None:
    """Stop before anything is written if a SOP system in the committed results cannot be re-run here.

    Without this, a missing embedding model makes sop_eval skip the hybrid system and report.write_all rewrites the
    README without the hybrid row."""
    p = results / "sop" / "sop_eval.json"
    committed = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if allow_drop or "hybrid" not in committed.get("systems", {}):
        return
    from mesq.sop import EMBED_MODEL, _load_embedder
    if _load_embedder() is None:
        raise SystemExit(
            f"[run_all] the committed results contain the hybrid SOP system, but {EMBED_MODEL} cannot be loaded "
            "(MESQ_NO_DENSE is set, sentence-transformers is missing, or the model is not in the local Hugging Face "
            "cache). Nothing was written. Download the model once (see README 'Reproduce') or pass --bm25-only to "
            "accept dropping the hybrid row.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-vision", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--bm25-only", action="store_true",
                    help="evaluate the SOP assistant without the embedding model (drops the hybrid row)")
    args = ap.parse_args(argv)
    if args.bm25_only:
        os.environ["MESQ_NO_DENSE"] = "1"
    check_sop_systems(RESULTS, allow_drop=args.bm25_only)

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
