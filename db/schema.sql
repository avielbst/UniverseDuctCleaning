-- ============================================================
-- DuctAI Copilot — schema.sql
-- Run once automatically on first docker compose up
-- Safe to re-run: uses CREATE TABLE IF NOT EXISTS
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT,
    email       TEXT,
    city        TEXT,
    state       TEXT,
    zip         TEXT,
    lead_source TEXT,
    created_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS employees (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    name_key    TEXT NOT NULL UNIQUE,  -- normalized: uppercase, trimmed
    pay_type    TEXT,                  -- 'commission' | 'hourly' | 'unknown'
    commission_rate NUMERIC(5,4)       -- e.g. 0.25 = 25%. NULL if hourly.
);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(id),   -- nullable: may not resolve
    employee_id     INTEGER REFERENCES employees(id), -- nullable: may not resolve
    status          TEXT,
    description     TEXT,
    address         TEXT,
    city            TEXT,
    state           TEXT,
    zip             TEXT,
    job_amount      NUMERIC(10,2),  -- what customer paid (after discount)
    subtotal        NUMERIC(10,2),  -- sum of line items (ML signal)
    discount_amount NUMERIC(10,2),  -- subtotal - job_amount
    discount_pct    NUMERIC(6,2),   -- discount as % of subtotal
    scheduled_at    TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS line_items (
    id          SERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL REFERENCES jobs(id),
    service     TEXT NOT NULL,          -- raw name from CSV
    service_key TEXT NOT NULL,          -- normalized: lowercase, canonical
    price       NUMERIC(10,2)
);

CREATE TABLE IF NOT EXISTS estimates (
    id          TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(id),
    employee_id INTEGER REFERENCES employees(id),
    status      TEXT,
    outcome     TEXT,   -- 'won' | 'lost' | 'open'
    won_value   NUMERIC(10,2),
    open_value  NUMERIC(10,2),
    lost_value  NUMERIC(10,2),
    lead_source TEXT,
    created_at  TIMESTAMPTZ,
    scheduled_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_jobs_customer_id   ON jobs(customer_id);
CREATE INDEX IF NOT EXISTS idx_jobs_employee_id   ON jobs(employee_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at    ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_zip           ON jobs(zip);
CREATE INDEX IF NOT EXISTS idx_line_items_job_id  ON line_items(job_id);
CREATE INDEX IF NOT EXISTS idx_line_items_key     ON line_items(service_key);
CREATE INDEX IF NOT EXISTS idx_estimates_customer ON estimates(customer_id);