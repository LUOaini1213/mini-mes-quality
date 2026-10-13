import os
import sys
from pathlib import Path

os.environ.setdefault("MESQ_NO_DENSE", "1")   # tests use BM25 only: deterministic and no model needed (CI)
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
