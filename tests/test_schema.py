"""The SQLite/DuckDB DDL and the Postgres DDL describe the same tables and columns."""
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "mesq"


def parse(path):
    text = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
    out = {}
    for m in re.finditer(r"CREATE TABLE (\w+) \((.*?)\n\);", text, re.S):
        cols = []
        for line in m.group(2).splitlines():
            line = line.strip()
            if line and not line.startswith("PRIMARY KEY"):
                cols.append(line.split()[0])
        out[m.group(1)] = cols
    return out


def test_same_tables_and_columns():
    a, b = parse(ROOT / "schema.sql"), parse(ROOT / "schema_postgres.sql")
    assert len(a) == 15
    assert a == b


def test_sqlite_schema_executes():
    con = sqlite3.connect(":memory:")
    con.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    names = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"lot_genealogy", "unit_lot", "inspection_result", "gt_cause"} <= names
