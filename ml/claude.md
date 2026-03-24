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

**Phase 1 results (2026-03-24, cutoff 2025-12-01)**:

| Label | AUROC | Baseline | Beats baseline | Predictor |
|-------|-------|----------|----------------|-----------|
| sanitation disinfectant | 0.8639 | 0.8433 | yes | lgbm |
| dryer vent cleaning | 0.9032 | 0.8986 | yes | lgbm |
| air duct deep cleaning | 0.9335 | 0.8865 | yes | lgbm |
| uv light system and installation | 0.8788 | 0.7204 | yes | lgbm |
| maintenance air duct cleaning | 0.8323 | 0.8323 | — | baseline |
| dryer vent additional length | 0.8505 | 0.8476 | yes | lgbm |
| duct encapsulation | 0.8559 | 0.8559 | — | baseline |
| dryer vent cleaning roof unclog | 0.8227 | 0.8227 | — | baseline |
| blower deep cleaning + coil | 0.8835 | 0.8771 | yes | lgbm |
| blower cleaning | 0.8612 | 0.8252 | yes | lgbm |

Mean AUROC: 0.8685 | All 10 labels > 0.65 | 7/10 beat baseline

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

**Phase 1 results (2026-03-24, cutoff 2025-12-01)**:

| Quantile | Pinball loss | Baseline | Improvement |
|----------|-------------|----------|-------------|
| P25 | 269.30 | 338.88 | +20.5% |
| P50 | 394.95 | 395.08 | ~0% |
| P75 | 387.42 | 451.27 | +14.1% |

Coverage: 44.9% | Mean range width: $985 | Feature importance: month 54%, city_median 26%

**Known limitations — do not use for customer-facing quotes yet**:

- **P50 median is no better than the naive baseline.** The model cannot predict the center
  of the price distribution. With 289 training rows and high price variance ($0–$11k),
  the signal-to-noise ratio is too low for meaningful median prediction.
- **Root cause is missing features, not model choice.** Price is driven by what services
  are performed and job complexity — neither is available at estimate time because line
  items don't exist yet. Month and city median are the only real signals available,
  and the model has found them (54% + 26% importance). Customer history features are
  essentially zero because they add noise, not signal, at this sample size.
- **Current deployment posture**: internal rough band only. Owner can use the range as
  a sanity check before quoting, not as a final quote. Never surface to customers.

**Planned redesign (Phase 2+)**:

Chain with the upsell model: given first_service + predicted upsell bundle → predict
job_amount from completed jobs (not estimates). This uses line_items data which contains
the real pricing drivers, and allows conditioning on service complexity. The upsell model
output becomes a feature for the pricing model, which is the correct product flow anyway.

---

## Model roles and usage

The two models serve **different users at different moments** in the workflow.

| Model | Primary user | Moment | Output |
|-------|-------------|--------|--------|
| Upsell classifier | Technician | Before arriving on site | Ranked service recommendations with probabilities |
| Pricing predictor | Owner | During estimate / quoting | Price range with confidence tier |

**Upsell classifier** — the technician opens the Streamlit UI on their phone before the appointment. The model surfaces ranked recommendations ("recommend Sanitation Disinfectant 78%, Blower Cleaning 61%"). The technician pitches these on site. Adoption is tracked in the dashboard.

**Pricing predictor** — when a new lead comes in, the owner asks the agent: *"Thumbtack lead in Boca Raton wants duct cleaning — what should I quote?"* The agent calls `PRICING_TOOL`, returns `{range: [$900, $1,400], median: $1,150, confidence: "medium"}`. The `city_rejection_rate` feature encodes price sensitivity per market.

Both are exposed as agent tools (`UPSELL_TOOL`, `PRICING_TOOL`) — the owner can invoke either via natural language. They are independent models that share features (lead source, customer history, city) but operate at different stages of the sales funnel.


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