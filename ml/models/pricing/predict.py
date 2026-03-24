"""
predict_price(job_profile) — inference function for the pricing predictor.

Usage:
    from ml.models.pricing.predict import predict_price

    result = predict_price({
        "lead_source":           "google",
        "is_returning_customer": False,
        "prior_avg_job_value":   0.0,
        "prior_job_count":       0,
        "city_median_job_value": 1350.0,
        "city_rejection_rate":   0.35,
        "month":                 3,
        "season":                "spring",
    })
"""
import json
import os
import pickle

import pandas as pd

ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "pricing"))

_models: dict | None = None
_meta:   dict | None = None


def _load():
    global _models, _meta
    if _meta is not None:
        return
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No trained model found at {meta_path}. Run train.py first.")
    with open(meta_path) as f:
        _meta = json.load(f)
    _models = {}
    for q in _meta["quantiles"]:
        label = f"p{int(q*100)}"
        with open(os.path.join(ARTIFACTS_DIR, f"{label}.pkl"), "rb") as f:
            _models[label] = pickle.load(f)


def _confidence_tier(job_profile: dict) -> tuple[str, int]:
    """
    Count training rows with matching lead_source and city_median_job_value within ±30%.
    Returns (confidence_tier, n_similar).
    """
    query_lead    = str(job_profile.get("lead_source", "unknown"))
    query_city    = float(job_profile.get("city_median_job_value", 0))
    lower, upper  = query_city * 0.70, query_city * 1.30

    n_similar = sum(
        1 for row in _meta["training_index"]
        if str(row["lead_source"]) == query_lead
        and lower <= float(row["city_median_job_value"]) <= upper
    )

    if n_similar >= 20:
        tier = "high"
    elif n_similar >= 10:
        tier = "medium"
    else:
        tier = "low"

    return tier, n_similar


def predict_price(job_profile: dict) -> dict:
    """
    Return a price range prediction for a given job/estimate profile.

    Args:
        job_profile: dict with keys:
            lead_source, is_returning_customer, prior_avg_job_value,
            prior_job_count, city_median_job_value, city_rejection_rate,
            month, season

    Returns:
        {
            "range":      [p25, p75],   # predicted price range
            "median":     p50,          # point estimate
            "confidence": str,          # "high" / "medium" / "low"
            "n_similar":  int,          # training samples used for confidence
        }
    """
    _load()

    feature_cols = _meta["feature_cols"]

    # Build 1-row DataFrame with sensible defaults for missing keys
    defaults = {
        "lead_source":           "unknown",
        "is_returning_customer": False,
        "prior_avg_job_value":   0.0,
        "prior_job_count":       0,
        "city_median_job_value": _meta["train_median"],
        "city_rejection_rate":   0.35,
        "month":                 6,
        "season":                "summer",
    }
    defaults.update(job_profile)
    row = pd.DataFrame([defaults])[feature_cols]

    for col in ["lead_source", "season"]:
        row[col] = row[col].astype("category")
    row["is_returning_customer"] = row["is_returning_customer"].astype(bool)

    p25 = float(_models["p25"].predict(row)[0])
    p50 = float(_models["p50"].predict(row)[0])
    p75 = float(_models["p75"].predict(row)[0])

    # Enforce monotonicity: p25 <= p50 <= p75
    p25 = min(p25, p50)
    p75 = max(p75, p50)

    confidence, n_similar = _confidence_tier(job_profile)

    return {
        "range":      [round(p25, 2), round(p75, 2)],
        "median":     round(p50, 2),
        "confidence": confidence,
        "n_similar":  n_similar,
    }


if __name__ == "__main__":
    import json as _json

    sample = {
        "lead_source":           "thumbtack",
        "is_returning_customer": False,
        "prior_avg_job_value":   0.0,
        "prior_job_count":       0,
        "city_median_job_value": 1350.0,
        "city_rejection_rate":   0.35,
        "month":                 3,
        "season":                "spring",
    }
    print(_json.dumps(predict_price(sample), indent=2))
