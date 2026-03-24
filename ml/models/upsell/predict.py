"""
predict_upsell(job_profile) — inference function for the upsell classifier.

For each service label, uses either the trained LightGBM model or the
P(label|first_service) baseline lookup, depending on which performed better
during training (stored in metadata.json).

Usage:
    from ml.models.upsell.predict import predict_upsell

    results = predict_upsell({
        "first_service":        "air duct deep cleaning",
        "lead_source":          "google",
        "is_returning_customer": False,
        "prior_job_count":       0,
        "prior_avg_job_value":   0.0,
        "city_median_job_value": 1350.0,
        "job_amount":            1200.0,
        "month":                 3,
    })
"""
import json
import os
import pickle

import pandas as pd

ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "upsell"))

# Lazy singleton — loaded once on first call
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
    for label in _meta["labels"]:
        if not _meta["label_stats"][label]["use_baseline"]:
            path = os.path.join(ARTIFACTS_DIR, f"{label.replace(' ', '_')}.pkl")
            with open(path, "rb") as f:
                _models[label] = pickle.load(f)


def _confidence_tier(n_train_pos: int) -> str:
    if n_train_pos >= 100:
        return "high"
    if n_train_pos >= 40:
        return "medium"
    return "low"


def predict_upsell(job_profile: dict) -> list[dict]:
    """
    Return ranked upsell service recommendations for a given job profile.

    Args:
        job_profile: dict with keys matching FEATURE_COLS from training:
            first_service, lead_source, is_returning_customer,
            prior_job_count, prior_avg_job_value,
            city_median_job_value, job_amount, month

    Returns:
        List of dicts sorted by probability (descending):
        [
            {
                "service":     str,   # service label name
                "probability": float, # model confidence (0-1)
                "recommend":   bool,  # True if probability >= precision threshold
                "confidence":  str,   # "high" / "medium" / "low" based on training support
            },
            ...
        ]
    """
    _load()

    feature_cols  = _meta["feature_cols"]
    first_service = job_profile.get("first_service", "")

    # Build a 1-row DataFrame; missing keys default to 0 / "unknown"
    defaults = {col: 0 for col in feature_cols}
    defaults.update({"first_service": "unknown", "lead_source": "unknown"})
    defaults.update(job_profile)
    row = pd.DataFrame([defaults])[feature_cols]

    for col in ["first_service", "lead_source"]:
        row[col] = row[col].astype("category")
    row["is_returning_customer"] = row["is_returning_customer"].astype(bool)

    recommendations = []

    for label in _meta["labels"]:
        stats        = _meta["label_stats"][label]
        thr          = stats["precision_threshold"]
        base_rates   = stats["baseline_rates"]
        base_fallback = stats["baseline_fallback"]
        confidence   = _confidence_tier(stats["n_train_pos"])

        if stats["use_baseline"]:
            prob = base_rates.get(first_service, base_fallback)
        else:
            row_l = row.copy()
            row_l["baseline_prob"] = base_rates.get(first_service, base_fallback)
            prob = float(_models[label].predict_proba(row_l)[:, 1][0])

        recommendations.append({
            "service":     label,
            "probability": round(prob, 4),
            "recommend":   prob >= thr,
            "confidence":  confidence,
        })

    return sorted(recommendations, key=lambda x: -x["probability"])


if __name__ == "__main__":
    import json as _json

    sample = {
        "first_service":         "air duct deep cleaning",
        "lead_source":           "google",
        "is_returning_customer": False,
        "prior_job_count":       0,
        "prior_avg_job_value":   0.0,
        "city_median_job_value": 1350.0,
        "job_amount":            1200.0,
        "month":                 3,
    }
    print(_json.dumps(predict_upsell(sample), indent=2))
