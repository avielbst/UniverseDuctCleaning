# DuctAI Copilot — Claude Code Project Memory

## What this project is

AI-powered sales and operations agent for Universe Duct Cleaning, a home
services company in South Florida. Built on real Jobber CRM data.

Three components:
1. **ML models** — upsell classifier + pricing predictor (Phase 1, active)
2. **LLM agent** — answers natural language business questions via SQL tool calls (Phase 2)
3. **Dashboard** — tracks upsell adoption rate and revenue impact (Phase 3)

Portfolio project + real deployed product for DS/AI/ML interviews.

---

## Current phase

**Phase 1 — ML Models** (active)

Two models to build:

| Model | Type | Target | Training data |
|-------|------|--------|---------------|
| Upsell classifier | Multi-label classification | Which services bought together in same job | 1,022 jobs with line items (Mar 2025–Mar 2026) |
| Pricing predictor | Quantile regression | Estimate value range for a given job profile | 419 won estimates |

Done when:
- [ ] Upsell model AUROC > 0.65 on 3+ service labels (time-based split)
- [ ] Pricing model outputs `{range: [low, high], confidence: medium/low}`
- [ ] Both models logged in MLflow with features, metrics, training date
- [ ] `predict_upsell(job_profile)` and `predict_price(job_profile)` functions return valid JSON
- [ ] Baseline comparison documented for both models

---

## Phase status

| Phase | Goal | Status |
|-------|------|--------|
| Phase 0 — Data foundation | All CRM data in PostgreSQL, 12 SQL views | ✅ Complete |
| Phase 1 — ML models | Upsell classifier + pricing predictor | 🔄 Active |
| Phase 2 — LLM agent | Natural language Q&A, grounded in tool calls | ⬜ Pending |
| Phase 3 — Deployment | Live on AWS, public URL | ⬜ Pending |
| Phase 4 — Impact measurement | Adoption rate tracked, revenue delta quantified | ⬜ Pending |

---

## Data model

**Always read `db/schema.sql` for full column definitions.**

Five tables. Load order (FK dependencies):

```
customers (1) ──< jobs (3) ──< line_items (4)
              └─< estimates (5)
employees (2) ──< jobs (3)
```

### Critical revenue facts

| Field | Meaning | Use for |
|-------|---------|---------|
| `subtotal` | Sum of line items at catalog price | ML signal — true service value |
| `job_amount` | What customer paid after discount | Revenue reporting |
| `discount_amount` | subtotal − job_amount | Discount analysis |
| `subtotal = NULL` | Job predates Mar 2025 (no service request) | Do not confuse with $0 job |
| `subtotal = 0` | Service request exists but free job | Keep, do not drop |

### Critical customer identity fact

**`customer_id` is NOT a reliable person identifier.**
Jobber creates a new `customer_id` when returning customers book via
pipeline automation or Thumbtack. 81 phone numbers are linked to 2+ IDs.

**Always use `canonical_id` from `v_customer_identity` when building
ML features that require customer history.** This deduplicates by phone
number, using the earliest record as the canonical person.

```sql
-- Wrong: undercounts returning customers
SELECT customer_id, COUNT(*) FROM jobs GROUP BY customer_id

-- Correct: groups all records for same person
SELECT ci.canonical_id, COUNT(*)
FROM jobs j
JOIN v_customer_identity ci ON ci.customer_id = j.customer_id
GROUP BY ci.canonical_id
```

---

## ML views (use these for feature building)

| View | Purpose |
|------|---------|
| `v_customer_history` | Per-person job stats using canonical_id: prior_job_count, prior_avg_job_value, is_returning_customer, last_job_date, lead_source |
| `v_city_price_index` | Median/P25/P75 job value per city (min 3 jobs). Cities below threshold → fall back to state median |
| `v_service_cooccurrence` | Which services appear together in the same job — upsell model foundation |

---

## Phase 1 — ML feature sets

### Upsell classifier features

Built from `line_items` joined to `jobs` and `v_customer_history`:

| Feature | Source | Notes |
|---------|--------|-------|
| `first_service` | line_items (min price or first parsed) | Service that initiated the job |
| `lead_source` | v_customer_history | Fills 97% null gap in estimates |
| `is_returning_customer` | v_customer_history | Based on canonical_id |
| `prior_job_count` | v_customer_history | Loyalty signal |
| `city_median_job_value` | v_city_price_index | Market price signal |
| `month` | jobs.created_at | Seasonality |

Target: binary vector of services purchased in same job (multi-label)
Training data: 1,022 jobs with line items (Mar 2025 – Mar 2026)
Split: time-based — train before 2025-12-01, test after

### Pricing predictor features

Built from `estimates` joined to `v_customer_history` and `v_city_price_index`:

| Feature | Source | Notes |
|---------|--------|-------|
| `lead_source` | v_customer_history (fills estimate gap) | Primary price sensitivity signal |
| `is_returning_customer` | v_customer_history | Trust-based pricing signal |
| `prior_avg_job_value` | v_customer_history | Anchor for repeat customers |
| `city_median_job_value` | v_city_price_index | Local market pricing |
| `month` | estimates.created_at | Florida seasonality |
| `season` | derived from month | summer/winter/spring/fall |

Target: `estimates.value` where `outcome = 'won'`
Training data: 419 won estimates
Model: LightGBM quantile regression (3 models: P25, P50, P75)
Output: `{"range": [low, high], "median": mid, "confidence": "medium"}`
Note: 419 samples is small — show confidence intervals, never bare point estimates

---

## ETL conventions

- All loaders in `etl/` accept `path` (filepath or data_dir)
- `find_file(data_dir, keyword)` in utils.py handles filename variants and `__failures_N` suffixes
- `run_all.py` passes specific filepath to each loader — do not change this
- Employees CSV is at `data/employees.csv` (not in data/raw) — load_employees ignores data_dir
- All inserts use before/after COUNT for accurate row tracking
- line_items uses truncate-reload (no natural PK)

---

## Infrastructure (local)

```
Service     Port   Credentials
────────────────────────────────────────────────
PostgreSQL  5432   ductai / ductai / ductai_db
MLflow      5000   (Phase 1+, no auth locally)
FastAPI     8000   (Phase 2+)
Streamlit   8501   (Phase 3+)
```

```bash
docker compose up -d           # start stack
docker compose down -v         # wipe DB and stop
python -m etl.run_all          # run full ETL pipeline
psql $DATABASE_URL             # connect to DB
```

---

## Known data issues (summary)

Full detail in `docs/data_notes.md`. Critical ones for ML:

| Issue | Impact | Mitigation |
|-------|--------|------------|
| customer_id not unique per person | Undercounts returning customers | Use canonical_id from v_customer_identity |
| Line items only from Mar 2025 | 1-year ML training window, not 3 years | Document, plan features accordingly |
| Service names inconsistent | ML label noise | normalize_service_name() in utils.py |
| "Express Job Closed" / "External Job Reminder" | Fake customers corrupt features | Filtered in v_customer_identity |
| 419 won estimates for pricing model | Thin training set, wide intervals | Use quantile regression + confidence flag |
| Lead source null for 42% of customers | Feature gap | Treat as 'unknown' category, never drop |
