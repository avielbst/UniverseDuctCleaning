"""
Train one LightGBM binary classifier per upsell target label.

Usage:
    python -m ml.models.upsell.train                        # rolling: holdout = last 90 days
    python -m ml.models.upsell.train --cutoff 2025-12-01   # fixed cutoff (Phase 1 benchmark)
"""
import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import precision_recall_curve, roc_auc_score

from ml.features.build_features import UPSELL_TARGET_SERVICES, load_upsell_features

logger = logging.getLogger(__name__)

HOLDOUT_DAYS     = 90    # default rolling holdout window
PRECISION_TARGET = 0.70  # minimum precision for a recommendation to be surfaced
ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "upsell"))

FEATURE_COLS = [
    "first_service", "lead_source",
    "is_returning_customer", "prior_job_count", "prior_avg_job_value",
    "city_median_job_value", "job_amount", "month",
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "verbose": -1,
    "random_state": 42,
}


def _baseline_auroc(rates: dict, fallback: float, test_df: pd.DataFrame, label: str) -> float:
    """AUROC of the P(label=1 | first_service) lookup baseline."""
    preds = test_df["first_service"].map(rates).fillna(fallback)
    if test_df[label].nunique() < 2:
        return float("nan")
    return roc_auc_score(test_df[label], preds)


def _precision_threshold(y_true, y_prob, target: float) -> float:
    """Lowest threshold achieving >= target precision. Falls back to max-precision threshold."""
    prec_arr, _, thresholds = precision_recall_curve(y_true, y_prob)
    mask = prec_arr[:-1] >= target
    if mask.any():
        return float(thresholds[mask][0])
    return float(thresholds[np.argmax(prec_arr[:-1])])


def train(cutoff: str | None = None, holdout_days: int = HOLDOUT_DAYS, refresh: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_upsell_features(refresh=refresh)

    if cutoff:
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    else:
        # Rolling window: hold out the most recent holdout_days of data
        cutoff_ts = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=holdout_days))
    train_df = df[df["job_date"] < cutoff_ts].copy()
    test_df  = df[df["job_date"] >= cutoff_ts].copy()
    logger.info("Cutoff: %s | Train: %d rows | Test: %d rows", cutoff_ts.date(), len(train_df), len(test_df))

    # Last 20% of training (time-ordered) used only for early stopping — not evaluation
    val_idx = int(len(train_df) * 0.8)
    tr      = train_df.iloc[:val_idx]
    val     = train_df.iloc[val_idx:]
    X_tr    = tr[FEATURE_COLS]
    X_val   = val[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]

    mlflow.set_experiment("upsell_classifier")
    with mlflow.start_run(run_name=f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params({
            "train_cutoff": str(cutoff_ts.date()),
            "holdout_days": holdout_days if not cutoff else "n/a (fixed cutoff)",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "features": ",".join(FEATURE_COLS),
            **{k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
        })

        models, results = {}, []

        for label in UPSELL_TARGET_SERVICES:
            if test_df[label].sum() == 0:
                logger.warning("Skipping %s — no positives in test set", label)
                continue

            # Baseline lookup computed from tr only — used as comparison AND as a feature
            fallback   = float(tr[label].mean())
            base_rates = tr.groupby("first_service", observed=True)[label].mean().to_dict()
            baseline   = _baseline_auroc(base_rates, fallback, test_df, label)

            # Add baseline_prob: target encoding of first_service for this label
            # Computed from tr, applied to all splits to avoid leakage
            X_tr_l   = X_tr.assign(baseline_prob=tr["first_service"].map(base_rates).fillna(fallback))
            X_val_l  = X_val.assign(baseline_prob=val["first_service"].map(base_rates).fillna(fallback))
            X_test_l = X_test.assign(baseline_prob=test_df["first_service"].map(base_rates).fillna(fallback))

            n_pos = int(tr[label].sum())
            n_neg = int((tr[label] == 0).sum())
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            model = LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
            model.fit(
                X_tr_l, tr[label],
                eval_set=[(X_val_l, val[label])],
                categorical_feature="auto",
                callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
            )

            y_prob       = model.predict_proba(X_test_l)[:, 1]
            auroc        = roc_auc_score(test_df[label], y_prob)
            prec_thr     = _precision_threshold(test_df[label], y_prob, PRECISION_TARGET)
            use_baseline = bool(auroc < baseline)

            short = label.replace(" ", "_")
            mlflow.log_metrics({
                f"auroc_{short}": round(auroc, 4),
                f"baseline_auroc_{short}": round(baseline, 4) if not np.isnan(baseline) else 0.0,
            })
            logger.info(
                "%-50s  auroc=%.4f  base=%.4f  thr=%.3f  use_baseline=%s",
                label, auroc, baseline, prec_thr, use_baseline,
            )

            models[label] = model
            results.append({
                "label":               label,
                "auroc":               auroc,
                "baseline_auroc":      baseline,
                "n_train_pos":         n_pos,
                "scale_pos_weight":    round(scale_pos_weight, 2),
                "precision_threshold": round(prec_thr, 4),
                "use_baseline":        use_baseline,
                "baseline_rates":      {str(k): float(v) for k, v in base_rates.items()},
                "baseline_fallback":   fallback,
            })

        aurocs = [r["auroc"] for r in results]
        above  = sum(a > 0.65 for a in aurocs)
        mlflow.log_metrics({"mean_auroc": round(np.mean(aurocs), 4), "labels_above_065": above})
        logger.info("Mean AUROC: %.4f | Labels > 0.65: %d/%d", np.mean(aurocs), above, len(results))

        for label, model in models.items():
            path = os.path.join(ARTIFACTS_DIR, f"{label.replace(' ', '_')}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

        label_stats = {
            r["label"]: {
                "n_train_pos":         r["n_train_pos"],
                "scale_pos_weight":    r["scale_pos_weight"],
                "precision_threshold": r["precision_threshold"],
                "use_baseline":        r["use_baseline"],
                "baseline_rates":      r["baseline_rates"],
                "baseline_fallback":   r["baseline_fallback"],
            }
            for r in results
        }
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
            json.dump({
                "feature_cols":   FEATURE_COLS,
                "precision_target": PRECISION_TARGET,
                "labels":         list(models.keys()),
                "label_stats":    label_stats,
                "train_cutoff":   str(cutoff_ts.date()),
                "trained_at":     datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        mlflow.log_artifacts(ARTIFACTS_DIR, artifact_path="models")
        logger.info("Artifacts saved to %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Train/test split date (YYYY-MM-DD). Defaults to today minus --holdout-days.")
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS,
                        help=f"Days to hold out as test set when no --cutoff is given (default: {HOLDOUT_DAYS}).")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-query features from DB, bypassing the parquet cache.")
    args = parser.parse_args()
    train(cutoff=args.cutoff, holdout_days=args.holdout_days, refresh=args.refresh)
