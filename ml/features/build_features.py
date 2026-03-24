"""
Builds feature DataFrames for the ML models by querying PostgreSQL DB
views.

Two public functions:
    build_upsell_features()  -> DataFrame for upsell classifier
    build_pricing_features() -> DataFrame for pricing predictor

Run standalone to verify DB views are working:
    uv run ./ml/features/build_features.py
"""
import logging
import sys
import os
import pandas as pd
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from etl.utils import get_engine

logger = logging.getLogger(__name__)


# Services with enough occurrences to be meaningful upsell labels
# Must have 30+ co-occurrences in training data to train a reliable classifier
UPSELL_TARGET_SERVICES = [
    "sanitation disinfectant",
    "dryer vent cleaning",
    "air duct deep cleaning",
    "uv light system and installation",
    "maintenance air duct cleaning",
    "dryer vent additional length",
    "duct encapsulation",
    "dryer vent cleaning roof unclog",
    "blower deep cleaning and coil maintenance",
    "blower cleaning",
]

# Month to season mapping
SEASON_MAP = {
    12: "winter", 1: "winter",  2: "winter",
    3:  "spring", 4: "spring",  5: "spring",
    6:  "summer", 7: "summer",  8: "summer",
    9:  "fall",   10: "fall",   11: "fall",
}


def _clean_lead_source(series: pd.Series) -> pd.Series:
    """
    Normalize lead source values to canonical categories.
    Groups low-frequency variants and fills nulls with 'unknown'.
    """
    mapping = {
        "Thumbtack":             "thumbtack",
        "Thumbtacks":            "thumbtack",
        "Funtack":               "thumbtack",
        "Google":                "google",
        "Google Guarantee":      "google",
        "Google Local Services": "google",
        "Google PPC":            "google",
        "referral":              "referral",
        "Facebook":              "social",
        "Instagram":             "social",
        "NextDoor App":          "social",
        "Neighbors App":         "social",
        "The Next Door App":     "social",
        "website":               "website",
        "Online":                "website",
    }
    return (
        series
        .map(mapping)
        .fillna("unknown")
        .astype("category")
    )


def _add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add month and season columns from a datetime column."""
    df["month"] = df[date_col].dt.month
    df["season"] = df["month"].map(SEASON_MAP).astype("category")
    return df

def build_upsell_features() -> pd.DataFrame:
    """
    Build the upsell classifier feature matrix.

    One row per job. Each target service gets its own binary column
    (1 = service purchased in this job, 0 = not purchased).

    Returns:
        DataFrame with columns:
            job_id, job_date,                          # identifiers + split key
            first_service, lead_source,                # categorical features
            is_returning_customer, prior_job_count,    # customer features
            city_median_job_value, job_amount, month,  # numeric features
            <service_label_1>, ..., <service_label_N>  # binary targets
    """
    engine = get_engine()

    # job + line items
    logger.info("Fetching jobs and line items...")
    jobs_df = pd.read_sql("""
        SELECT
            j.id            AS job_id,
            j.created_at    AS job_date,
            j.job_amount,
            j.city,
            j.state,
            j.customer_id
        FROM jobs j
        WHERE j.status = 'Completed'
          AND j.created_at IS NOT NULL
        ORDER BY j.created_at
    """, engine)

    li_df = pd.read_sql("""
        SELECT job_id, service_key, price
        FROM line_items
        ORDER BY job_id, id          -- id preserves CRM insertion order
    """, engine)

    # City price index
    logger.info("Fetching city price index...")
    city_df = pd.read_sql("""
        SELECT city, state, median_job_value AS city_median_job_value
        FROM v_city_price_index
    """, engine)

    # State-level fallback for cities with < 3 jobs - take state median instead.
    state_df = pd.read_sql("""
        SELECT state,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY job_amount) AS state_median
        FROM jobs
        WHERE status = 'Completed' AND job_amount > 0 AND state IS NOT NULL
        GROUP BY state
    """, engine)

    # Customer history
    logger.info("Fetching customer history...")
    cust_df = pd.read_sql("""
        SELECT
            ci.customer_id,
            ch.prior_job_count,
            ch.prior_avg_job_value,
            ch.is_returning_customer,
            ch.lead_source
        FROM v_customer_identity ci
        JOIN v_customer_history ch ON ch.canonical_id = ci.canonical_id
    """, engine)

    # Derive first_service per job
    # First non-zero priced item in CRM's insertion order = primary service
    logger.info("Deriving first_service per job...")
    li_nonzero = li_df[li_df["price"] > 0].copy()
    first_svc = (
        li_nonzero
        .groupby("job_id")["service_key"]
        .first()
        .reset_index()
        .rename(columns={"service_key": "first_service"})
    )

    # Build binary target columns
    logger.info("Building service target label columns...")
    li_targets = li_df[li_df["service_key"].isin(UPSELL_TARGET_SERVICES)].copy()
    li_targets["value"] = 1
    targets = (
        li_targets
        .pivot_table(
            index="job_id",
            columns="service_key",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    # Ensure all target columns exist even if service never appears
    for svc in UPSELL_TARGET_SERVICES:
        if svc not in targets.columns:
            targets[svc] = 0

    # Merge everything
    logger.info("Merging all features...")
    df = jobs_df.copy()
    df = df.merge(first_svc, on="job_id", how="left")
    df = df.merge(targets, on="job_id", how="left")
    df = df.merge(cust_df, on="customer_id", how="left")
    df = df.merge(city_df, on=["city", "state"], how="left")
    df = df.merge(state_df, on="state", how="left")

    # City median fallback to state median
    df["city_median_job_value"] = df["city_median_job_value"].fillna(
        df["state_median"]
    )

    # Clean and type
    df["job_date"] = pd.to_datetime(df["job_date"], utc=True)
    df = _add_time_features(df, "job_date")

    df["lead_source"] = _clean_lead_source(df["lead_source"])
    df["first_service"] = df["first_service"].fillna("unknown").astype("category")

    df["prior_job_count"] = df["prior_job_count"].fillna(0).clip(upper=10).astype(int)
    df["prior_avg_job_value"] = df["prior_avg_job_value"].fillna(0)
    df["is_returning_customer"] = df["is_returning_customer"].fillna(False)
    df["job_amount"] = df["job_amount"].fillna(0)
    df["city_median_job_value"] = df["city_median_job_value"].fillna(
        df["job_amount"].median()
    )

    # Fill target nulls with 0 (service not purchased)
    for svc in UPSELL_TARGET_SERVICES:
        if svc in df.columns:
            df[svc] = df[svc].fillna(0).astype(int)

    # Drop rows with no first_service -can't train on them
    before = len(df)
    df = df[df["first_service"] != "unknown"]
    logger.info("Dropped %d jobs with no parseable first service", before - len(df))

    feature_cols = [
        "job_id", "job_date",
        "first_service", "lead_source",
        "is_returning_customer", "prior_job_count", "prior_avg_job_value",
        "city_median_job_value", "job_amount", "month",
    ] + UPSELL_TARGET_SERVICES

    df = df[[c for c in feature_cols if c in df.columns]]

    logger.info(
        "Upsell feature matrix: %d rows x %d cols", len(df), len(df.columns)
    )
    return df.sort_values("job_date").reset_index(drop=True)


def build_pricing_features() -> pd.DataFrame:
    """
    Build the pricing predictor feature matrix.
 
    One row per won estimate. Target is the estimate value.
 
    Returns:
        DataFrame with columns:
            estimate_id, estimate_date,                # identifiers + split key
            lead_source, is_returning_customer,        # customer features
            prior_avg_job_value,                       # customer history
            city_median_job_value,                     # market signal
            month, season,                             # time features
            value                                      # target
    """
    engine = get_engine()
 
    logger.info("Fetching won estimates...")
    est_df = pd.read_sql("""
        SELECT
            e.id            AS estimate_id,
            e.created_at    AS estimate_date,
            e.value
        FROM estimates e
        WHERE e.outcome = 'won'
        AND e.value > 0
        AND e.created_at IS NOT NULL    
        ORDER BY e.created_at
    """, engine)
 
    logger.info("Fetching customer history for estimates...")
    cust_df = pd.read_sql("""
        SELECT
            e.id                                        AS estimate_id,
            COUNT(j.id) > 0                             AS is_returning_customer,
            COUNT(j.id)                                 AS prior_job_count,
            ROUND(AVG(j.job_amount)::NUMERIC, 2)        AS prior_avg_job_value,
            MAX(ci.lead_source)                         AS lead_source,
            MAX(ci.city)                                AS city,
            MAX(ci.state)                               AS state
        FROM estimates e
        JOIN v_customer_identity ci ON ci.customer_id = e.customer_id
        LEFT JOIN jobs j ON j.customer_id = ci.customer_id
            AND j.status = 'Completed'
            AND j.job_amount > 0
            AND j.completed_at < e.created_at     -- only jobs BEFORE this estimate
        WHERE e.outcome = 'won'
        AND e.value > 0
        AND e.created_at IS NOT NULL
        GROUP BY e.id
    """, engine)
 
    logger.info("Fetching city price index...")
    city_df = pd.read_sql("""
        SELECT city, state, median_job_value AS city_median_job_value
        FROM v_city_price_index
    """, engine)
 
    state_df = pd.read_sql("""
        SELECT state,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY job_amount) AS state_median
        FROM jobs
        WHERE status = 'Completed' AND job_amount > 0 AND state IS NOT NULL
        GROUP BY state
    """, engine)
 
    # City rejection rate from lost estimates
    # Encodes price sensitivity per market using signal from lost estimates.
    # Cities with fewer than 3 decided estimates use the overall fallback rate.
    logger.info("Fetching city rejection rates from estimates...")
    rejection_df = pd.read_sql("""
        SELECT
            c.city,
            COUNT(*)                                             AS total_decided,
            SUM(CASE WHEN e.outcome = 'lost' THEN 1 ELSE 0 END) AS lost_count,
            ROUND(
                SUM(CASE WHEN e.outcome = 'lost' THEN 1 ELSE 0 END)::NUMERIC /
                NULLIF(COUNT(*), 0)
            , 3)                                                 AS city_rejection_rate
        FROM estimates e
        JOIN customers c ON c.id = e.customer_id
        WHERE e.outcome IN ('won', 'lost')
          AND c.city IS NOT NULL
        GROUP BY c.city
        HAVING COUNT(*) >= 3
    """, engine)
 
    # Overall rejection rate - fallback for cities with < 3 estimates
    overall_rejection = float(pd.read_sql("""
        SELECT ROUND(
            SUM(CASE WHEN outcome = 'lost' THEN 1 ELSE 0 END)::NUMERIC /
            NULLIF(COUNT(*), 0)
        , 3) AS overall_rejection_rate
        FROM estimates
        WHERE outcome IN ('won', 'lost')
    """, engine).iloc[0, 0])
 
    logger.info(
        "City rejection rates for %d cities. Overall fallback: %.2f",
        len(rejection_df), overall_rejection
    )
 
    # merge
    df = est_df.merge(cust_df, on="estimate_id", how="left")
    df = df.merge(city_df, on=["city", "state"], how="left")
    df = df.merge(state_df, on="state", how="left")
    df = df.merge(rejection_df[["city", "city_rejection_rate"]], on="city", how="left")
 
    df["city_median_job_value"] = df["city_median_job_value"].fillna(
        df["state_median"]
    )
    # Cities below the 3-estimate threshold get the overall rejection rate
    df["city_rejection_rate"] = df["city_rejection_rate"].fillna(overall_rejection)
 
    # Clean and type
    df["estimate_date"] = pd.to_datetime(df["estimate_date"], utc=True)
    df = _add_time_features(df, "estimate_date")
 
    df["lead_source"] = _clean_lead_source(df["lead_source"])
    df["is_returning_customer"] = df["is_returning_customer"].fillna(False)
    df["prior_avg_job_value"] = df["prior_avg_job_value"].fillna(0)
    df["prior_job_count"] = df["prior_job_count"].fillna(0).clip(upper=10).astype(int)
    df["city_median_job_value"] = df["city_median_job_value"].fillna(
        df["value"].median()
    )
 
    feature_cols = [
        "estimate_id", "estimate_date",
        "lead_source", "is_returning_customer",
        "prior_avg_job_value", "prior_job_count",
        "city_median_job_value",
        "city_rejection_rate",
        "month", "season",
        "value",
    ]
 
    df = df[[c for c in feature_cols if c in df.columns]]
 
    logger.info(
        "Pricing feature matrix: %d rows x %d cols", len(df), len(df.columns)
    )
    return df.sort_values("estimate_date").reset_index(drop=True)
 
 
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))


def load_upsell_features(refresh: bool = False) -> pd.DataFrame:
    """Return upsell feature matrix, loading from parquet cache when available."""
    path = os.path.join(DATA_DIR, "upsell_features.parquet")
    if not refresh and os.path.exists(path):
        logger.info("Loading upsell features from cache: %s", path)
        df = pd.read_parquet(path)
        for col in ["first_service", "lead_source", "season"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df
    df = build_upsell_features()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Upsell features cached to %s", path)
    return df


def load_pricing_features(refresh: bool = False) -> pd.DataFrame:
    """Return pricing feature matrix, loading from parquet cache when available."""
    path = os.path.join(DATA_DIR, "pricing_features.parquet")
    if not refresh and os.path.exists(path):
        logger.info("Loading pricing features from cache: %s", path)
        df = pd.read_parquet(path)
        for col in ["lead_source", "season"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df
    df = build_pricing_features()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Pricing features cached to %s", path)
    return df


# for debugging.

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
 
    print("\n" + "="*50)
    print("UPSELL FEATURES")
    print("="*50)
    upsell = build_upsell_features()
    print(f"Shape: {upsell.shape}")
    print(f"\nDate range: {upsell['job_date'].min().date()} → {upsell['job_date'].max().date()}")
    print(f"\nFeature dtypes:\n{upsell.dtypes.to_string()}")
    print(f"\nTarget label distribution:")
    for svc in UPSELL_TARGET_SERVICES:
        if svc in upsell.columns:
            pct = upsell[svc].mean() * 100
            print(f"  {pct:5.1f}%  {svc}")
    print(f"\nNull counts:\n{upsell.isnull().sum()[upsell.isnull().sum()>0].to_string()}")
 
    print("\n" + "="*50)
    print("PRICING FEATURES")
    print("="*50)
    pricing = build_pricing_features()
    print(f"Shape: {pricing.shape}")
    print(f"\nDate range: {pricing['estimate_date'].min().date()} → {pricing['estimate_date'].max().date()}")
    print(f"\nTarget (value) distribution:")
    print(pricing["value"].describe().round(0).to_string())
    print(f"\nLead source distribution:\n{pricing['lead_source'].value_counts().to_string()}")
    print(f"\nReturning customers: {pricing['is_returning_customer'].mean()*100:.1f}%")
    print(f"\nNull counts:\n{pricing.isnull().sum()[pricing.isnull().sum()>0].to_string()}")
 