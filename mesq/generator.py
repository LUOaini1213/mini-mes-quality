"""Synthetic MES data generator with injected, known root causes.

EVERYTHING THIS MODULE PRODUCES IS SYNTHETIC. No real factory, line, product, supplier or person is modelled.

The simulated line builds one printed-circuit-board assembly (PCBA) through five steps:

    10 PRINT   solder-paste printing      PRN-01, PRN-02     consumes a SOLDER_PASTE batch and a PCB_BLANK batch
    20 PLACE   pick-and-place             PNP-01..PNP-03
    30 REFLOW  reflow oven                RFL-01, RFL-02
    40 AOI     automated optical insp.    AOI-01, AOI-02     (inspection results are written later by mesq.link)
    50 ASSY    module assembly            ASM-01

Lots are released per work order, may be SPLIT after PLACE (children go through REFLOW/AOI separately) and the
children are MERGED back before ASSY. Every unit therefore has exactly one lot that carries each step, but the lot
that carries a step is generally not the unit's final lot: tracing a unit's history needs the genealogy.

Ground truth (written to gt_cause / gt_unit, never read by the analysis):

* DRIFT: PNP-02 drifts on the night shift (C) from day ``drift_start_day`` on; the added defect probability ramps
  linearly from 0 to ``drift_max_add`` at the end of the horizon.
* BAD_BATCH: solder-paste batch SP-B07 adds ``bad_batch_add`` to the defect probability of every unit printed with it.

Deliberate traps (no effect on defects, but correlated with a cause, so a naive one-factor-at-a-time analysis
can flag them):

* Operator OP-009 (night shift) is assigned to PNP-02 most of the time -> confounded with DRIFT.
* RFL-02 is down for maintenance while SP-B07 is in use on PRN-01, so those lots go to RFL-01 -> confounded with
  BAD_BATCH.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

T0 = datetime(2026, 1, 5, 6, 0)  # synthetic calendar start (a Monday, start of shift A)
STEPS = [(10, "PRINT", "Solder paste printing"), (20, "PLACE", "Pick and place"), (30, "REFLOW", "Reflow soldering"),
         (40, "AOI", "Automated optical inspection"), (50, "ASSY", "Module assembly")]
EQUIPMENT = {"PRINT": ["PRN-01", "PRN-02"], "PLACE": ["PNP-01", "PNP-02", "PNP-03"], "REFLOW": ["RFL-01", "RFL-02"],
             "AOI": ["AOI-01", "AOI-02"], "ASSY": ["ASM-01"]}
SHIFTS = "ABC"
PRODUCT = "PCBA-X1"
DRIFT_KEY = "PLACE.equipment=PNP-02 & PLACE.shift=C"
BAD_BATCH_KEY = "PRINT.material=SP-B07"
TRAP_KEYS = {"PLACE.operator=OP-009": DRIFT_KEY, "REFLOW.equipment=RFL-01": BAD_BATCH_KEY}


@dataclasses.dataclass
class GenConfig:
    seed: int = 7
    days: int = 42
    wo_per_day: float = 2.5          # mean work orders released per day
    wo_qty: tuple = (48, 144)        # units per work order (uniform, multiple of 8)
    max_lot: int = 96
    split_prob: float = 0.45
    base_defect: float = 0.02
    lot_noise_sd: float = 0.004      # lot-to-lot noise on the base rate
    drift_start_day: int = 10
    drift_max_add: float = 0.22
    bad_batch: str = "SP-B07"
    bad_batch_add: float = 0.12
    bad_batch_hours: float = 60.0
    bad_window_prn01_share: float = 0.7
    trap_operator: str = "OP-009"
    trap_operator_share: float = 0.7


def shift_of(ts: datetime) -> str:
    h = ts.hour
    return "A" if 6 <= h < 14 else ("B" if 14 <= h < 22 else "C")


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


class _Gen:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.end = T0 + timedelta(days=cfg.days)
        self.rows = {k: [] for k in ["product", "route_step", "equipment", "operator", "material_batch", "work_order",
                                      "lot", "lot_genealogy", "lot_step", "material_consumption", "unit", "unit_lot",
                                      "gt_unit", "gt_cause"]}
        self.lot_seq = 0
        self.unit_seq = 0
        self.unit_lots: dict[str, list[str]] = {}
        self.lot_units: dict[str, list[str]] = {}
        self.unit_prob: dict[str, float] = {}

    # ---- master data -------------------------------------------------------------------------------------------
    def master(self):
        r = self.rows
        r["product"].append(dict(product_id=PRODUCT, description="Synthetic PCB assembly (illustrative only)"))
        for seq, code, desc in STEPS:
            r["route_step"].append(dict(product_id=PRODUCT, step_seq=seq, step_code=code, description=desc))
        for code, eqs in EQUIPMENT.items():
            for e in eqs:
                r["equipment"].append(dict(equipment_id=e, step_code=code, line_id="L1"))
        # 4 operators per shift (pseudonymous). OP-001..004 -> A, 005..008 -> B, 009..012 -> C
        self.ops = {s: [f"OP-{i:03d}" for i in range(1 + 4 * k, 5 + 4 * k)] for k, s in enumerate(SHIFTS)}
        for s, ops in self.ops.items():
            for o in ops:
                r["operator"].append(dict(operator_id=o, home_shift=s))
        # Solder paste: each printer consumes its own sequence of batches, ~2.3 days each.
        # SP-B07 is pinned to PRN-01 in a window around day 12-15 (see _paste_batch).
        self.paste_windows = []  # (printer, batch, start, end)
        n = 1
        for p in EQUIPMENT["PRINT"]:
            t = T0
            while t < self.end:
                dur = timedelta(hours=float(self.rng.uniform(44, 68)))
                if p == "PRN-01" and t <= T0 + timedelta(days=12) < t + dur:
                    bid = self.cfg.bad_batch  # the bad batch is in use across the day-12 boundary on PRN-01
                    dur = timedelta(hours=self.cfg.bad_batch_hours)
                else:
                    if f"SP-B{n:02d}" == self.cfg.bad_batch:
                        n += 1  # the bad batch ID is reserved for the window above
                    bid = f"SP-B{n:02d}"
                    n += 1
                self.paste_windows.append((p, bid, t, t + dur))
                t += dur
        batches = sorted({w[1] for w in self.paste_windows})
        for b in batches:
            start = min(w[2] for w in self.paste_windows if w[1] == b)
            r["material_batch"].append(dict(batch_id=b, material_type="SOLDER_PASTE",
                                            supplier_code="SUP-A" if int(b[-2:]) % 2 else "SUP-B",
                                            received_date=(start - timedelta(days=3)).strftime("%Y-%m-%d")))
        for i in range(self.cfg.days // 5 + 1):
            r["material_batch"].append(dict(batch_id=f"PB-{i + 1:02d}", material_type="PCB_BLANK",
                                            supplier_code="SUP-C", received_date=(T0 + timedelta(days=5 * i - 2)).strftime("%Y-%m-%d")))
        bad = [w for w in self.paste_windows if w[1] == self.cfg.bad_batch]
        assert len(bad) == 1, "generator invariant: the bad batch occupies exactly one window"
        self.bad_window = bad[0]
        r["gt_cause"].append(dict(cause_key=DRIFT_KEY, added_defect_prob=self.cfg.drift_max_add,
                                  description=f"PNP-02 drifts on night shift C from day {self.cfg.drift_start_day} "
                                              f"(added defect probability ramps 0 -> {self.cfg.drift_max_add})"))
        r["gt_cause"].append(dict(cause_key=BAD_BATCH_KEY, added_defect_prob=self.cfg.bad_batch_add,
                                  description=f"Solder paste batch {self.cfg.bad_batch} is bad (+{self.cfg.bad_batch_add})"))

    def _paste_batch(self, printer, ts):
        for p, b, s, e in self.paste_windows:
            if p == printer and s <= ts < e:
                return b
        return [w for w in self.paste_windows if w[0] == printer][-1][1]

    # ---- lots ---------------------------------------------------------------------------------------------------
    def new_lot(self, wo, qty, origin, ts, units):
        self.lot_seq += 1
        lot_id = f"L{self.lot_seq:05d}"
        self.rows["lot"].append(dict(lot_id=lot_id, work_order_id=wo, qty=qty, origin=origin, created_ts=_iso(ts),
                                     status="ACTIVE"))
        self.lot_units[lot_id] = list(units)
        for u in units:
            self.unit_lots[u].append(lot_id)
        return lot_id

    def set_status(self, lot_id, status):
        for row in reversed(self.rows["lot"]):
            if row["lot_id"] == lot_id:
                row["status"] = status
                return

    def run_step(self, lot_id, seq, code, ts):
        cfg, rng = self.cfg, self.rng
        ts = ts + timedelta(minutes=float(rng.uniform(10, 120)))  # queue time
        shift = shift_of(ts)
        eqs = EQUIPMENT[code]
        day = (ts - T0).total_seconds() / 86400
        if code == "REFLOW" and self.bad_window[2] <= ts < self.bad_window[3]:
            eq = "RFL-01"  # trap: RFL-02 down for maintenance while SP-B07 is in use
        elif code == "PRINT" and self.bad_window[2] <= ts < self.bad_window[3]:
            # PRN-01 takes most of the work while SP-B07 is loaded (keeps the bad batch's exposure from being
            # left to chance; tests/test_generator.py checks the exposure on every seed used in the results)
            eq = "PRN-01" if rng.random() < cfg.bad_window_prn01_share else "PRN-02"
        else:
            eq = eqs[int(rng.integers(len(eqs)))]
        ops = self.ops[shift]
        if code == "PLACE" and shift == "C" and eq == "PNP-02" and rng.random() < cfg.trap_operator_share:
            op = cfg.trap_operator  # trap: OP-009 usually runs PNP-02 at night (but also other machines)
        else:
            op = ops[int(rng.integers(len(ops)))]
        n = len(self.lot_units[lot_id])
        dur = timedelta(minutes=float(n * rng.uniform(0.4, 0.8) + 15))
        out = ts + dur
        self.rows["lot_step"].append(dict(lot_id=lot_id, step_seq=seq, equipment_id=eq, operator_id=op, shift=shift,
                                          track_in_ts=_iso(ts), track_out_ts=_iso(out)))
        add = 0.0
        if code == "PRINT":
            paste = self._paste_batch(eq, ts)
            blank = f"PB-{int(day // 5) + 1:02d}"
            self.rows["material_consumption"].append(dict(lot_id=lot_id, step_seq=seq, batch_id=paste,
                                                          qty_used=round(n * 0.85, 2)))
            self.rows["material_consumption"].append(dict(lot_id=lot_id, step_seq=seq, batch_id=blank, qty_used=float(n)))
            if paste == cfg.bad_batch:
                add += cfg.bad_batch_add
        if code == "PLACE" and eq == "PNP-02" and shift == "C" and day >= cfg.drift_start_day:
            add += cfg.drift_max_add * (day - cfg.drift_start_day) / (cfg.days - cfg.drift_start_day)
        for u in self.lot_units[lot_id]:
            self.unit_prob[u] += add
        return out

    def work_order(self, k, release):
        cfg, rng = self.cfg, self.rng
        wo = f"WO-{k:04d}"
        qty = int(rng.integers(cfg.wo_qty[0] // 8, cfg.wo_qty[1] // 8 + 1)) * 8
        self.rows["work_order"].append(dict(work_order_id=wo, product_id=PRODUCT, qty_planned=qty, release_ts=_iso(release)))
        n_lots = int(np.ceil(qty / cfg.max_lot))
        sizes = [qty // n_lots + (1 if i < qty % n_lots else 0) for i in range(n_lots)]
        for size in sizes:
            units = []
            for _ in range(size):
                self.unit_seq += 1
                u = f"U{self.unit_seq:06d}"
                units.append(u)
                self.unit_lots[u] = []
                self.unit_prob[u] = 0.0
            lot = self.new_lot(wo, size, "RELEASE", release, units)
            lot_noise = float(rng.normal(0, cfg.lot_noise_sd))
            for u in units:
                self.unit_prob[u] = max(0.0, cfg.base_defect + lot_noise)
            t = release
            t = self.run_step(lot, 10, "PRINT", t)
            t = self.run_step(lot, 20, "PLACE", t)
            if size >= 16 and rng.random() < cfg.split_prob:
                cut = int(rng.integers(size // 4, 3 * size // 4 + 1))
                parts = [units[:cut], units[cut:]]
                children = []
                for part in parts:
                    c = self.new_lot(wo, len(part), "SPLIT", t, part)
                    self.rows["lot_genealogy"].append(dict(parent_lot_id=lot, child_lot_id=c, relation="SPLIT",
                                                           qty=len(part), event_ts=_iso(t)))
                    children.append(c)
                self.set_status(lot, "CONSUMED")
                ends = []
                for c in children:
                    tc = self.run_step(c, 30, "REFLOW", t)
                    tc = self.run_step(c, 40, "AOI", tc)
                    ends.append(tc)
                tm = max(ends)
                merged = self.new_lot(wo, size, "MERGE", tm, units)
                for c, part in zip(children, parts):
                    self.rows["lot_genealogy"].append(dict(parent_lot_id=c, child_lot_id=merged, relation="MERGE",
                                                           qty=len(part), event_ts=_iso(tm)))
                    self.set_status(c, "CONSUMED")
                self.run_step(merged, 50, "ASSY", tm)
                self.set_status(merged, "COMPLETE")
            else:
                t = self.run_step(lot, 30, "REFLOW", t)
                t = self.run_step(lot, 40, "AOI", t)
                self.run_step(lot, 50, "ASSY", t)
                self.set_status(lot, "COMPLETE")

    def run(self):
        self.master()
        k = 0
        t = T0
        while True:
            t += timedelta(hours=float(self.rng.exponential(24 / self.cfg.wo_per_day)))
            if t >= self.end - timedelta(hours=12):
                break
            k += 1
            self.work_order(k, t)
        r = self.rows
        for u, lots in self.unit_lots.items():
            r["unit"].append(dict(unit_id=u, final_lot_id=lots[-1]))
            for lot in lots:
                r["unit_lot"].append(dict(unit_id=u, lot_id=lot))
            p = min(1.0, self.unit_prob[u])
            r["gt_unit"].append(dict(unit_id=u, true_defect=int(self.rng.random() < p), defect_prob=round(p, 6)))
        return {k: pd.DataFrame(v) for k, v in r.items()}


def generate(cfg: GenConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return the synthetic MES tables as DataFrames (keys = table names in schema.sql)."""
    return _Gen(cfg or GenConfig()).run()
