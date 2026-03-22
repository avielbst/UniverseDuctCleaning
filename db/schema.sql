-- ============================================================
-- DuctAI Copilot — Database Schema
-- Source of truth for all table definitions.
-- Runs automatically on first docker compose up via initdb.d.
-- Safe to re-run: uses CREATE TABLE IF NOT EXISTS.
-- ============================================================


-- ------------------------------------------------------------
-- customers
-- Source: data/raw/*customer*.csv
-- PK: Jobber native ID (stable, unique per person)
-- ------------------------------------------------------------
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


-- ------------------------------------------------------------
-- employees
-- Source: data/employees.csv (manually maintained by owner)
--
-- pay_type values:
--   'commission' — flat % of job_amount
--   'tiered'     — % depends on job_amount thresholds (e.g. Matthew)
--   'hourly'     — fixed $/hr (e.g. Jibril)
--   'salary'     — fixed $/month (e.g. Aimee)
--   'hybrid'     — hourly + commission (e.g. Matt)
--   'unknown'    — not yet confirmed with owner
--
-- role values:
--   'owner'      — business owner, 0% commission regardless
--   'technician' — field worker
--   'office'     — office/admin staff
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS employees (
    id                          SERIAL PRIMARY KEY,
    name                        TEXT NOT NULL,
    name_key                    TEXT NOT NULL UNIQUE,  -- normalized uppercase for ETL matching
    role                        TEXT,
    pay_type                    TEXT,
    commission_rate             NUMERIC(5,4),          -- base rate e.g. 0.30 = 30%
    commission_tier1_rate       NUMERIC(5,4),          -- tier 1 rate (Matthew: 0.18)
    commission_tier1_threshold  NUMERIC(10,2),         -- sales threshold for tier 1
    commission_tier2_rate       NUMERIC(5,4),          -- tier 2 rate (Matthew: 0.20)
    commission_tier2_threshold  NUMERIC(10,2),         -- sales threshold for tier 2
    hourly_rate                 NUMERIC(8,2),          -- $/hr for hourly/hybrid employees
    monthly_salary              NUMERIC(10,2)          -- $/month for salaried employees
);


-- ------------------------------------------------------------
-- jobs
-- Source: data/raw/*jobs*.csv + data/raw/*service_request*.csv
--
-- Revenue fields:
--   job_amount      = what customer paid (after discount)
--   subtotal        = sum of line items (full catalog price) — ML signal
--   discount_amount = subtotal - job_amount
--   discount_pct    = discount as % of subtotal
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(id),    -- nullable: name may not resolve
    employee_id     INTEGER REFERENCES employees(id), -- nullable: tag may not parse
    status          TEXT,
    description     TEXT,
    address         TEXT,
    city            TEXT,
    state           TEXT,
    zip             TEXT,
    job_amount      NUMERIC(10,2),
    subtotal        NUMERIC(10,2),
    discount_amount NUMERIC(10,2),
    discount_pct    NUMERIC(6,2),
    scheduled_at    TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ,
    notes           TEXT
);


-- ------------------------------------------------------------
-- line_items
-- Source: parsed from service_requests Line Items text column
-- One row per service per job — primary ML training signal.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS line_items (
    id          SERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL REFERENCES jobs(id),
    service     TEXT NOT NULL,      -- raw name from CSV
    service_key TEXT NOT NULL,      -- normalized lowercase canonical label
    price       NUMERIC(10,2)
);


-- ------------------------------------------------------------
-- estimates
-- Source: data/raw/*estimates*.csv
-- Potential revenue not yet realized — used for conversion analysis.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS estimates (
    id           TEXT PRIMARY KEY,
    customer_id  TEXT REFERENCES customers(id),
    status       TEXT,
    outcome      TEXT,       -- 'won' | 'lost' | 'open'
    value        NUMERIC(10,2),  -- the relevant value for this outcome
    lead_source  TEXT,
    created_at   TIMESTAMPTZ,
    scheduled_at TIMESTAMPTZ
);


-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_jobs_customer_id   ON jobs(customer_id);
CREATE INDEX IF NOT EXISTS idx_jobs_employee_id   ON jobs(employee_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at    ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_zip           ON jobs(zip);
CREATE INDEX IF NOT EXISTS idx_line_items_job_id  ON line_items(job_id);
CREATE INDEX IF NOT EXISTS idx_line_items_key     ON line_items(service_key);
CREATE INDEX IF NOT EXISTS idx_estimates_customer ON estimates(customer_id);