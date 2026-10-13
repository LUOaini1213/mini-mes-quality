-- mini-mes-quality: MES-style data model (SQLite / DuckDB dialect).
-- ALL DATA LOADED INTO THIS SCHEMA BY THIS PROJECT IS SYNTHETIC.
-- A Postgres-compatible version lives in schema_postgres.sql; keep the two in sync
-- (tests/test_schema.py checks that both declare the same tables and columns).

CREATE TABLE product (
    product_id      TEXT PRIMARY KEY,
    description     TEXT NOT NULL
);

CREATE TABLE route_step (               -- the process route (one route per product)
    product_id      TEXT NOT NULL REFERENCES product(product_id),
    step_seq        INTEGER NOT NULL,       -- 10, 20, 30 ...
    step_code       TEXT NOT NULL,          -- PRINT, PLACE, REFLOW, AOI, ASSY
    description     TEXT NOT NULL,
    PRIMARY KEY (product_id, step_seq)
);

CREATE TABLE equipment (
    equipment_id    TEXT PRIMARY KEY,       -- e.g. PNP-02
    step_code       TEXT NOT NULL,          -- which kind of step it can run
    line_id         TEXT NOT NULL
);

CREATE TABLE operator (                 -- pseudonymous IDs only, no names
    operator_id     TEXT PRIMARY KEY,       -- e.g. OP-007
    home_shift      TEXT NOT NULL           -- A / B / C
);

CREATE TABLE material_batch (
    batch_id        TEXT PRIMARY KEY,       -- e.g. SP-B07
    material_type   TEXT NOT NULL,          -- SOLDER_PASTE, PCB_BLANK, ...
    supplier_code   TEXT NOT NULL,          -- pseudonymous (SUP-A ...)
    received_date   TEXT NOT NULL
);

CREATE TABLE work_order (
    work_order_id   TEXT PRIMARY KEY,
    product_id      TEXT NOT NULL REFERENCES product(product_id),
    qty_planned     INTEGER NOT NULL,
    release_ts      TEXT NOT NULL
);

CREATE TABLE lot (
    lot_id          TEXT PRIMARY KEY,
    work_order_id   TEXT NOT NULL REFERENCES work_order(work_order_id),
    qty             INTEGER NOT NULL,
    origin          TEXT NOT NULL,          -- RELEASE | SPLIT | MERGE
    created_ts      TEXT NOT NULL,
    status          TEXT NOT NULL           -- ACTIVE | CONSUMED (split/merged away) | COMPLETE
);

CREATE TABLE lot_genealogy (            -- parent -> child edges for split and merge
    parent_lot_id   TEXT NOT NULL REFERENCES lot(lot_id),
    child_lot_id    TEXT NOT NULL REFERENCES lot(lot_id),
    relation        TEXT NOT NULL,          -- SPLIT | MERGE
    qty             INTEGER NOT NULL,       -- units moved along this edge
    event_ts        TEXT NOT NULL,
    PRIMARY KEY (parent_lot_id, child_lot_id)
);

CREATE TABLE lot_step (                 -- what happened to a lot at a step (track-in / track-out)
    lot_id          TEXT NOT NULL REFERENCES lot(lot_id),
    step_seq        INTEGER NOT NULL,
    equipment_id    TEXT NOT NULL REFERENCES equipment(equipment_id),
    operator_id     TEXT NOT NULL REFERENCES operator(operator_id),
    shift           TEXT NOT NULL,          -- A 06-14, B 14-22, C 22-06
    track_in_ts     TEXT NOT NULL,
    track_out_ts    TEXT NOT NULL,
    PRIMARY KEY (lot_id, step_seq)
);

CREATE TABLE material_consumption (
    lot_id          TEXT NOT NULL REFERENCES lot(lot_id),
    step_seq        INTEGER NOT NULL,
    batch_id        TEXT NOT NULL REFERENCES material_batch(batch_id),
    qty_used        REAL NOT NULL,
    PRIMARY KEY (lot_id, step_seq, batch_id)
);

CREATE TABLE unit (                     -- serialised unit (one board)
    unit_id         TEXT PRIMARY KEY,
    final_lot_id    TEXT NOT NULL REFERENCES lot(lot_id)
);

CREATE TABLE unit_lot (                 -- every lot a unit has been a member of
    unit_id         TEXT NOT NULL REFERENCES unit(unit_id),
    lot_id          TEXT NOT NULL REFERENCES lot(lot_id),
    PRIMARY KEY (unit_id, lot_id)
);

CREATE TABLE inspection_result (        -- written by the inspection step (AOI)
    unit_id         TEXT NOT NULL REFERENCES unit(unit_id),
    step_seq        INTEGER NOT NULL,
    equipment_id    TEXT NOT NULL REFERENCES equipment(equipment_id),
    image_ref       TEXT,                   -- dataset-relative image path (images are never stored in the repo)
    anomaly_score   REAL,
    decision        TEXT NOT NULL,          -- PASS | FAIL
    model_version   TEXT NOT NULL,
    PRIMARY KEY (unit_id, step_seq)
);

-- Ground truth of the synthetic generator. Kept in separate tables so that the traceability analysis
-- (mesq/trace.py) never reads them; only the scorer does.
CREATE TABLE gt_unit (
    unit_id         TEXT PRIMARY KEY REFERENCES unit(unit_id),
    true_defect     INTEGER NOT NULL,       -- 0 / 1
    defect_prob     REAL NOT NULL
);

CREATE TABLE gt_cause (
    cause_key       TEXT PRIMARY KEY,       -- e.g. "PLACE.equipment=PNP-02 & PLACE.shift=C"
    description     TEXT NOT NULL,
    added_defect_prob REAL NOT NULL
);
