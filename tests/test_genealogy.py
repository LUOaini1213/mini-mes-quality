"""Genealogy / traceability queries on a hand-built fixture and invariants on generated data."""
import pandas as pd
import pytest

from mesq import db
from mesq.generator import GenConfig, generate


def fixture_tables():
    T = {
        "product": pd.DataFrame([dict(product_id="P", description="x")]),
        "route_step": pd.DataFrame([dict(product_id="P", step_seq=s, step_code=c, description=c)
                                    for s, c in [(10, "PRINT"), (20, "PLACE"), (30, "REFLOW"), (40, "AOI"), (50, "ASSY")]]),
        "equipment": pd.DataFrame([dict(equipment_id=e, step_code=c, line_id="L1") for e, c in
                                   [("PRN-01", "PRINT"), ("PNP-01", "PLACE"), ("RFL-01", "REFLOW"), ("RFL-02", "REFLOW"),
                                    ("AOI-01", "AOI"), ("ASM-01", "ASSY")]]),
        "operator": pd.DataFrame([dict(operator_id="OP-001", home_shift="A")]),
        "material_batch": pd.DataFrame([dict(batch_id="SP-X", material_type="SOLDER_PASTE", supplier_code="S",
                                             received_date="2026-01-01"),
                                        dict(batch_id="SP-Y", material_type="SOLDER_PASTE", supplier_code="S",
                                             received_date="2026-01-01")]),
        "work_order": pd.DataFrame([dict(work_order_id="W1", product_id="P", qty_planned=4, release_ts="t"),
                                    dict(work_order_id="W2", product_id="P", qty_planned=2, release_ts="t")]),
        # A (4 units) -> split B (u1,u2) + C (u3,u4) -> merge D; E is unrelated (W2)
        "lot": pd.DataFrame([dict(lot_id=l, work_order_id=w, qty=q, origin=o, created_ts="t", status=s) for l, w, q, o, s in
                             [("A", "W1", 4, "RELEASE", "CONSUMED"), ("B", "W1", 2, "SPLIT", "CONSUMED"),
                              ("C", "W1", 2, "SPLIT", "CONSUMED"), ("D", "W1", 4, "MERGE", "COMPLETE"),
                              ("E", "W2", 2, "RELEASE", "COMPLETE")]]),
        "lot_genealogy": pd.DataFrame([dict(parent_lot_id=p, child_lot_id=c, relation=r, qty=2, event_ts="t") for p, c, r in
                                       [("A", "B", "SPLIT"), ("A", "C", "SPLIT"), ("B", "D", "MERGE"), ("C", "D", "MERGE")]]),
        "lot_step": pd.DataFrame([dict(lot_id=l, step_seq=s, equipment_id=e, operator_id="OP-001", shift="A",
                                       track_in_ts=f"2026-01-01 0{s // 10}:00:00", track_out_ts="t") for l, s, e in
                                  [("A", 10, "PRN-01"), ("A", 20, "PNP-01"), ("B", 30, "RFL-01"), ("C", 30, "RFL-02"),
                                   ("B", 40, "AOI-01"), ("C", 40, "AOI-01"), ("D", 50, "ASM-01"),
                                   ("E", 10, "PRN-01"), ("E", 20, "PNP-01"), ("E", 30, "RFL-02"), ("E", 40, "AOI-01"),
                                   ("E", 50, "ASM-01")]]),
        "material_consumption": pd.DataFrame([dict(lot_id="A", step_seq=10, batch_id="SP-X", qty_used=1.0),
                                              dict(lot_id="E", step_seq=10, batch_id="SP-Y", qty_used=1.0)]),
        "unit": pd.DataFrame([dict(unit_id=u, final_lot_id=l) for u, l in
                              [("u1", "D"), ("u2", "D"), ("u3", "D"), ("u4", "D"), ("u5", "E"), ("u6", "E")]]),
        "unit_lot": pd.DataFrame([dict(unit_id=u, lot_id=l) for u, l in
                                  [("u1", "A"), ("u1", "B"), ("u1", "D"), ("u2", "A"), ("u2", "B"), ("u2", "D"),
                                   ("u3", "A"), ("u3", "C"), ("u3", "D"), ("u4", "A"), ("u4", "C"), ("u4", "D"),
                                   ("u5", "E"), ("u6", "E")]]),
        "inspection_result": pd.DataFrame([dict(unit_id=u, step_seq=40, equipment_id="AOI-01", image_ref=None,
                                                anomaly_score=0.0, decision=d, model_version="m") for u, d in
                                           [("u1", "PASS"), ("u2", "PASS"), ("u3", "FAIL"), ("u4", "FAIL"),
                                            ("u5", "PASS"), ("u6", "PASS")]]),
    }
    return T


@pytest.fixture()
def con():
    c = db.build(fixture_tables())
    yield c
    c.close()


def test_ancestors_through_merge_and_split(con):
    assert set(db.ancestors(con, "D")) == {"D", "B", "C", "A"}
    assert db.ancestors(con, "D")[0] == "D"
    assert set(db.ancestors(con, "B")) == {"B", "A"}
    assert db.ancestors(con, "E") == ["E"]


def test_descendants(con):
    assert set(db.descendants(con, "A")) == {"A", "B", "C", "D"}
    assert set(db.descendants(con, "C")) == {"C", "D"}


def test_where_used_forward_trace(con):
    w = db.where_used(con, "SP-X")
    assert sorted(w.unit_id) == ["u1", "u2", "u3", "u4"]
    assert set(w.final_lot_id) == {"D"}
    assert list(db.where_used(con, "SP-Y").unit_id) == ["u5", "u6"]


def test_unit_history_backward_trace(con):
    h = db.unit_history(con, "u3")
    assert list(h.step_seq) == [10, 20, 30, 40, 50]
    assert list(h.lot_id) == ["A", "A", "C", "C", "D"]
    assert h.set_index("step_code").loc["REFLOW", "equipment_id"] == "RFL-02"
    assert db.unit_history(con, "u1").set_index("step_code").loc["REFLOW", "equipment_id"] == "RFL-01"


def test_factor_table_uses_genealogy(con):
    U = db.unit_factor_table(con)
    assert U.loc["u3", "REFLOW.equipment"] == "RFL-02" and U.loc["u1", "REFLOW.equipment"] == "RFL-01"
    assert U.loc["u3", "PRINT.material.SOLDER_PASTE"] == "SP-X"
    assert U["fail"].tolist() == [0, 0, 1, 1, 0, 0]


def test_foreign_keys_enforced():
    T = fixture_tables()
    T["lot_step"] = pd.concat([T["lot_step"], pd.DataFrame([dict(lot_id="A", step_seq=30, equipment_id="NOPE",
                                                                  operator_id="OP-001", shift="A", track_in_ts="t",
                                                                  track_out_ts="t")])])
    with pytest.raises(Exception):
        db.build(T)


@pytest.fixture(scope="module")
def gen():
    T = generate(GenConfig(seed=3, days=21))
    return T, db.build(T)


def test_each_unit_has_exactly_one_lot_per_step(gen):
    T, con = gen
    n = con.execute("""SELECT COUNT(*) FROM (SELECT ul.unit_id, ls.step_seq, COUNT(*) c FROM unit_lot ul
                       JOIN lot_step ls ON ls.lot_id = ul.lot_id GROUP BY 1, 2 HAVING c != 1)""").fetchone()[0]
    assert n == 0
    per_unit = con.execute("""SELECT COUNT(DISTINCT ls.step_seq) FROM unit_lot ul JOIN lot_step ls
                              ON ls.lot_id = ul.lot_id GROUP BY ul.unit_id""").fetchall()
    assert {r[0] for r in per_unit} == {5}


def test_split_and_merge_quantities_balance(gen):
    T, con = gen
    lot = T["lot"].set_index("lot_id")
    g = T["lot_genealogy"]
    for rel, side, other in [("SPLIT", "parent_lot_id", "child_lot_id"), ("MERGE", "child_lot_id", "parent_lot_id")]:
        s = g[g.relation == rel].groupby(side)["qty"].sum()
        assert len(s) > 0
        for lot_id, q in s.items():
            assert lot.loc[lot_id, "qty"] == q
        for _, e in g[g.relation == rel].iterrows():
            assert lot.loc[e[other], "qty"] == e["qty"]


def test_unit_lot_consistent_with_genealogy(gen):
    T, con = gen
    for u in T["unit"].unit_id.sample(50, random_state=0):
        final = T["unit"].set_index("unit_id").loc[u, "final_lot_id"]
        lots = set(T["unit_lot"][T["unit_lot"].unit_id == u].lot_id)
        assert lots <= set(db.ancestors(con, final))
