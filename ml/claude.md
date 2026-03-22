# ML Module — Claude Code Context

## Purpose

Phase 1: two LightGBM models trained on CRM data.
Read root CLAUDE.md first for data model and critical facts.

---

## Two models

### 1. Upsell classifier (`models/upsell/`)

**Problem**: Given a job profile, predict which additional services the
customer is likely to buy in the same job.

**Type**: Multi-label classification (one binary classifier per service label)

**Training data**:
- 1,026 jobs with parsed line items (March 2025 – March 2026)
- Average 2.9 services per job
- 75% of jobs have 2+ services — strong co-occurrence signal

**Split strategy**: Time-based cross-validation
- Do NOT use random split — that leaks future into past
- Use TimeSeriesSplit(n_splits=5) on jobs ordered by date
- Final evaluation on held-out jobs after 2025-12-01 (63 jobs)

**Target labels** (services with 30+ occurrences — enough signal):
- sanitation disinfectant
- dryer vent cleaning
- air duct deep cleaning
- uv light system and installation
- maintenance air duct cleaning
- dryer vent additional length
- duct encapsulation
- dryer vent cleaning roof unclog
- blower deep cleaning and coil maintenance
- blower cleaning

**Features**:

| Feature | Source | How to build |
|---------|--------|-------------|
| `first_service` | line_items (lowest price item) | Categorical, one-hot encode |
| `lead_source` | v_customer_history.lead_source | Fills 97% null in SR; 'unknown' if still null |
| `is_returning_customer` | v_customer_history | Bool via canonical_id |
| `prior_job_count` | v_customer_history | Int, clip at 10 |
| `city_median_job_value` | v_city_price_index | Float; state median fallback if city < 3 jobs |
| `month` | jobs.created_at | Int 1-12 |
| `job_amount` | jobs.job_amount | Float, 0 if null |

**Baseline to beat**:
Most-frequent co-occurring service per first_service.
e.g. for dryer vent cleaning → "dryer vent additional length" (139 co-occurrences)
Log baseline AUROC before training LightGBM.

**Done when**:
- AUROC > 0.65 on held-out test set for at least 3 service labels
- `predict_upsell(job_profile)` returns ranked list with probabilities
- MLflow run logged with all features, metrics, training date

---

### 2. Pricing predictor (`models/pricing/`)

**Problem**: Given a job/estimate profile, predict the price range that
maximizes both win rate and revenue.

**Type**: Quantile regression — train 3 separate LightGBM models:
- P25 model (alpha=0.25) → lower bound
- P50 model (alpha=0.50) → median / point estimate
- P75 model (alpha=0.75) → upper bound

Output: `{"range": [p25, p75], "median": p50, "confidence": "medium"}`

**Training data**:
- 419 won estimates (May 2023 – March 2026)
- Value distribution: min=$0, P25=$887, median=$1,350, P75=$1,943, max=$11,080
- Split: train before 2025-12-01 (292), test after (127)

**Critical**: Lead source is only 2% populated in estimates directly.
ALWAYS join to customers table via customer name to get lead source.
Use v_customer_history for all customer-level features.

**Features**:

| Feature | Source | How to build |
|---------|--------|-------------|
| `lead_source` | v_customer_history (NOT estimates.lead_source) | Categorical; 'unknown' if null |
| `is_returning_customer` | v_customer_history via canonical_id | Bool |
| `prior_avg_job_value` | v_customer_history | Float; 0 if new customer |
| `city_median_job_value` | v_city_price_index | Float; state median fallback |
| `month` | estimates.created_at | Int 1-12 |
| `season` | derived from month | 'summer'/'winter'/'spring'/'fall' |

**Confidence scoring**:
Count training samples within similar feature range.
- >= 20 similar samples → 'high'
- 10-19 → 'medium'
- < 10 → 'low'
Always surface confidence in output. Never return a range without it.

**Done when**:
- Pinball loss on test set better than naive median baseline
- `predict_price(job_profile)` returns `{range, median, confidence, n_similar}`
- MLflow run logged

---

## Conventions

- Always import `get_connection` from `etl.utils` — never open psycopg2 directly
- Build features from DB views, never re-read raw CSVs
- Log every training run to MLflow: features used, hyperparams, metrics, date, row count
- Model artifacts saved to `ml/artifacts/` (gitignored)
- All predict functions return JSON-serializable dicts
- Use `canonical_id` from `v_customer_identity`, never raw `customer_id` for person-level features

---

## Running order

```bash
# 1. Build features (verify DB views work)
python -m ml.features.build_features

# 2. Train upsell model
python -m ml.models.upsell.train

# 3. Evaluate upsell model
python -m ml.models.upsell.evaluate

# 4. Train pricing model
python -m ml.models.pricing.train

# 5. Evaluate pricing model
python -m ml.models.pricing.evaluate
```