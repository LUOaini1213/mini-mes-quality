"""Tracked text files must not contain machine-specific absolute paths (drive letters, user home folders)."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Built from pieces so that this file does not match itself.
DRIVE = r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\\\|\\|/)(?!/)[A-Za-z0-9_.$~-]"
HOME = r"(?:[Cc]:[\\/]+Users|/home|/Users)[\\/]+[A-Za-z0-9_.-]+"
PATTERN = re.compile(DRIVE + "|" + HOME)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".jsonl", ".csv", ".sql", ".ini", ".toml", ".cfg",
                 ".html", ".sha256", ""}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
        return [ROOT / f for f in out.decode("utf-8").split("\0") if f]
    except (OSError, subprocess.CalledProcessError):     # not a git checkout (e.g. a source archive)
        return [p for p in ROOT.rglob("*") if p.is_file() and not {".git", ".venv", "build", "data"} & set(p.parts)]


def find_local_paths(files):
    hits = []
    for p in files:
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    return hits


def test_pattern_catches_local_paths_and_ignores_urls():
    bad = ["D:" + "\\data\\visa\\raw", "r\"D:" + "\\\\data\\\\visa\"", "E:" + "/tiktok/x", "C:" + "/Users/someone/x",
           "/home" + "/someone/project"]
    good = ["https://example.com/a", "file:///tmp", "a:b", "ratio 1:2", "s3://bucket/key", "data/visa/raw"]
    assert all(PATTERN.search(s) for s in bad), [s for s in bad if not PATTERN.search(s)]
    assert not any(PATTERN.search(s) for s in good), [s for s in good if PATTERN.search(s)]


def test_no_local_paths_in_tracked_text_files():
    hits = find_local_paths(tracked_files())
    assert not hits, "machine-specific paths in tracked files:\n" + "\n".join(hits)
