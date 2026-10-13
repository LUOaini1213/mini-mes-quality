"""Load the (synthetic) MES tables into SQLite and run genealogy / traceability queries.

SQLite is used because it ships with Python; the same SQL (recursive CTEs, window-free) also runs on DuckDB and
Postgres. See schema.sql / schema_postgres.sql.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = Path(__file__).with_name("schema.sql")
TABLE_ORDER = ["product", "route_step", "equipment", "operator", "material_batch", "work_order", "lot",
               "lot_genealogy", "lot_step", "material_consumption", "unit", "unit_lot", "inspection_result",
               "gt_unit", "gt_cause"]


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA.read_text(encoding="utf-8"))


def load(con: sqlite3.Connection, tables: dict[str, pd.DataFrame]) -> None:
    for name in TABLE_ORDER:
        df = tables.get(name)
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        con.executemany(f"INSERT INTO {name} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        df.itertuples(index=False, name=None))
    con.commit()


def build(tables: dict[str, pd.DataFrame], path: str | Path = ":memory:") -> sqlite3.Connection:
    p = Path(path) if path != ":memory:" else None
    if p is not None and p.exists():
        p.unlink()
    con = connect(path)
    create_schema(con)
    load(con, tables)
    return con


# ---- genealogy ----------------------------------------------------------------------------------------------------

ANCESTORS_SQL = """
WITH RECURSIVE anc(lot_id, depth) AS (
    SELECT ?, 0
    UNION
    SELECT g.parent_lot_id, a.depth + 1 FROM lot_genealogy g JOIN anc a ON g.child_lot_id = a.lot_id
)
SELECT lot_id, MIN(depth) AS depth FROM anc GROUP BY lot_id ORDER BY depth, lot_id
"""

DESCENDANTS_SQL = """
WITH RECURSIVE des(lot_id, depth) AS (
    SELECT ?, 0
    UNION
    SELECT g.child_lot_id, d.depth + 1 FROM lot_genealogy g JOIN des d ON g.parent_lot_id = d.lot_id
)
SELECT lot_id, MIN(depth) AS depth FROM des GROUP BY lot_id ORDER BY depth, lot_id
"""


def ancestors(con, lot_id: str) -> list[str]:
    """All lots upstream of lot_id (including itself), via split/merge edges."""
    return [r[0] for r in con.execute(ANCESTORS_SQL, (lot_id,))]


def descendants(con, lot_id: str) -> list[str]:
    """All lots downstream of lot_id (including itself)."""
    return [r[0] for r in con.execute(DESCENDANTS_SQL, (lot_id,))]


def where_used(con, batch_id: str) -> pd.DataFrame:
    """Forward trace: every final (COMPLETE) lot and unit that contains material from batch_id."""
    sql = """
    WITH RECURSIVE hit(lot_id) AS (
        SELECT DISTINCT lot_id FROM material_consumption WHERE batch_id = ?
        UNION
        SELECT g.child_lot_id FROM lot_genealogy g JOIN hit h ON g.parent_lot_id = h.lot_id
    )
    SELECT DISTINCT u.unit_id, u.final_lot_id
    FROM unit u JOIN unit_lot ul ON ul.unit_id = u.unit_id
    WHERE ul.lot_id IN (SELECT lot_id FROM material_consumption WHERE batch_id = ?)
      AND u.final_lot_id IN (SELECT lot_id FROM hit)
    ORDER BY u.unit_id
    """
    return pd.read_sql_query(sql, con, params=(batch_id, batch_id))


def unit_history(con, unit_id: str) -> pd.DataFrame:
    """Backward trace for one unit: for each route step, the lot that carried the unit, equipment, operator, shift."""
    sql = """
    SELECT ls.step_seq, rs.step_code, ls.lot_id, ls.equipment_id, ls.operator_id, ls.shift, ls.track_in_ts
    FROM unit_lot ul
    JOIN lot_step ls ON ls.lot_id = ul.lot_id
    JOIN lot l ON l.lot_id = ul.lot_id
    JOIN work_order w ON w.work_order_id = l.work_order_id
    JOIN route_step rs ON rs.product_id = w.product_id AND rs.step_seq = ls.step_seq
    WHERE ul.unit_id = ?
    ORDER BY ls.step_seq
    """
    return pd.read_sql_query(sql, con, params=(unit_id,))


def unit_factor_table(con) -> pd.DataFrame:
    """One row per unit with its as-processed path: <STEP>.equipment / .operator / .shift for every step, the
    material batches consumed, and the inspection decision. Built through unit_lot, so split/merge are honoured.
    Ground-truth tables are NOT read here."""
    steps = pd.read_sql_query(
        """SELECT ul.unit_id, rs.step_code, ls.equipment_id, ls.operator_id, ls.shift, ls.track_in_ts
           FROM unit_lot ul JOIN lot_step ls ON ls.lot_id = ul.lot_id
           JOIN lot l ON l.lot_id = ul.lot_id JOIN work_order w ON w.work_order_id = l.work_order_id
           JOIN route_step rs ON rs.product_id = w.product_id AND rs.step_seq = ls.step_seq""", con)
    wide = steps.pivot(index="unit_id", columns="step_code", values=["equipment_id", "operator_id", "shift"])
    short = {"equipment_id": "equipment", "operator_id": "operator", "shift": "shift"}
    wide.columns = [f"{step}.{short[f]}" for f, step in wide.columns]
    mats = pd.read_sql_query(
        """SELECT ul.unit_id, rs.step_code, mb.material_type, mc.batch_id
           FROM unit_lot ul JOIN material_consumption mc ON mc.lot_id = ul.lot_id
           JOIN material_batch mb ON mb.batch_id = mc.batch_id
           JOIN lot l ON l.lot_id = ul.lot_id JOIN work_order w ON w.work_order_id = l.work_order_id
           JOIN route_step rs ON rs.product_id = w.product_id AND rs.step_seq = mc.step_seq""", con)
    mats["col"] = mats["step_code"] + ".material." + mats["material_type"]
    mwide = mats.pivot(index="unit_id", columns="col", values="batch_id")
    insp = pd.read_sql_query("SELECT unit_id, decision, anomaly_score FROM inspection_result", con).set_index("unit_id")
    out = wide.join(mwide).join(insp, how="inner")
    out["fail"] = (out.pop("decision") == "FAIL").astype(int)
    first = steps[steps.step_code == "PRINT"].set_index("unit_id")["track_in_ts"]
    out["day"] = ((pd.to_datetime(first) - pd.to_datetime(first).min()).dt.total_seconds() // 86400).astype(int)
    return out.sort_index()
