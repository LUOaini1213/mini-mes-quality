-- mini-mes-quality: Postgres-compatible DDL for the same MES data model as schema.sql.
-- ALL DATA LOADED INTO THIS SCHEMA BY THIS PROJECT IS SYNTHETIC.
-- Timestamps are TIMESTAMP here (TEXT ISO-8601 in the SQLite/DuckDB file).

CREATE TABLE product (
    product_id      VARCHAR(32) PRIMARY KEY,
    description     TEXT NOT NULL
);

CREATE TABLE route_step (
    product_id      VARCHAR(32) NOT NULL REFERENCES product(product_id),
    step_seq        INTEGER NOT NULL,
    step_code       VARCHAR(16) NOT NULL,
    description     TEXT NOT NULL,
    PRIMARY KEY (product_id, step_seq)
);

CREATE TABLE equipment (
    equipment_id    VARCHAR(32) PRIMARY KEY,
    step_code       VARCHAR(16) NOT NULL,
    line_id         VARCHAR(16) NOT NULL
);

CREATE TABLE operator (
    operator_id     VARCHAR(16) PRIMARY KEY,
    home_shift      CHAR(1) NOT NULL CHECK (home_shift IN ('A', 'B', 'C'))
);

CREATE TABLE material_batch (
    batch_id        VARCHAR(32) PRIMARY KEY,
    material_type   VARCHAR(32) NOT NULL,
    supplier_code   VARCHAR(16) NOT NULL,
    received_date   DATE NOT NULL
);

CREATE TABLE work_order (
    work_order_id   VARCHAR(32) PRIMARY KEY,
    product_id      VARCHAR(32) NOT NULL REFERENCES product(product_id),
    qty_planned     INTEGER NOT NULL CHECK (qty_planned > 0),
    release_ts      TIMESTAMP NOT NULL
);

CREATE TABLE lot (
    lot_id          VARCHAR(32) PRIMARY KEY,
    work_order_id   VARCHAR(32) NOT NULL REFERENCES work_order(work_order_id),
    qty             INTEGER NOT NULL CHECK (qty > 0),
    origin          VARCHAR(8) NOT NULL CHECK (origin IN ('RELEASE', 'SPLIT', 'MERGE')),
    created_ts      TIMESTAMP NOT NULL,
    status          VARCHAR(10) NOT NULL CHECK (status IN ('ACTIVE', 'CONSUMED', 'COMPLETE'))
);

CREATE TABLE lot_genealogy (
    parent_lot_id   VARCHAR(32) NOT NULL REFERENCES lot(lot_id),
    child_lot_id    VARCHAR(32) NOT NULL REFERENCES lot(lot_id),
    relation        VARCHAR(8) NOT NULL CHECK (relation IN ('SPLIT', 'MERGE')),
    qty             INTEGER NOT NULL CHECK (qty > 0),
    event_ts        TIMESTAMP NOT NULL,
    PRIMARY KEY (parent_lot_id, child_lot_id)
);

CREATE TABLE lot_step (
    lot_id          VARCHAR(32) NOT NULL REFERENCES lot(lot_id),
    step_seq        INTEGER NOT NULL,
    equipment_id    VARCHAR(32) NOT NULL REFERENCES equipment(equipment_id),
    operator_id     VARCHAR(16) NOT NULL REFERENCES operator(operator_id),
    shift           CHAR(1) NOT NULL CHECK (shift IN ('A', 'B', 'C')),
    track_in_ts     TIMESTAMP NOT NULL,
    track_out_ts    TIMESTAMP NOT NULL,
    PRIMARY KEY (lot_id, step_seq)
);

CREATE TABLE material_consumption (
    lot_id          VARCHAR(32) NOT NULL REFERENCES lot(lot_id),
    step_seq        INTEGER NOT NULL,
    batch_id        VARCHAR(32) NOT NULL REFERENCES material_batch(batch_id),
    qty_used        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (lot_id, step_seq, batch_id)
);

CREATE TABLE unit (
    unit_id         VARCHAR(32) PRIMARY KEY,
    final_lot_id    VARCHAR(32) NOT NULL REFERENCES lot(lot_id)
);

CREATE TABLE unit_lot (
    unit_id         VARCHAR(32) NOT NULL REFERENCES unit(unit_id),
    lot_id          VARCHAR(32) NOT NULL REFERENCES lot(lot_id),
    PRIMARY KEY (unit_id, lot_id)
);

CREATE TABLE inspection_result (
    unit_id         VARCHAR(32) NOT NULL REFERENCES unit(unit_id),
    step_seq        INTEGER NOT NULL,
    equipment_id    VARCHAR(32) NOT NULL REFERENCES equipment(equipment_id),
    image_ref       TEXT,
    anomaly_score   DOUBLE PRECISION,
    decision        VARCHAR(4) NOT NULL CHECK (decision IN ('PASS', 'FAIL')),
    model_version   VARCHAR(64) NOT NULL,
    PRIMARY KEY (unit_id, step_seq)
);

CREATE TABLE gt_unit (
    unit_id         VARCHAR(32) PRIMARY KEY REFERENCES unit(unit_id),
    true_defect     SMALLINT NOT NULL CHECK (true_defect IN (0, 1)),
    defect_prob     DOUBLE PRECISION NOT NULL
);

CREATE TABLE gt_cause (
    cause_key       TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    added_defect_prob DOUBLE PRECISION NOT NULL
);

CREATE INDEX ix_genealogy_child ON lot_genealogy (child_lot_id);
CREATE INDEX ix_unit_lot_lot ON unit_lot (lot_id);
CREATE INDEX ix_consumption_batch ON material_consumption (batch_id);
