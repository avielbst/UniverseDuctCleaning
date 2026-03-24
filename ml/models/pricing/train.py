"""
Train 3 LightGBM quantile regression models (P25, P50, P75) for job price prediction.

Hyperparameters are tuned via Optuna on the P50 (median) model using time-series
cross-validation, then reused for P25 and P75.

Usage:
    python -m ml.models.pricing.train                          # rolling 90-day holdout
    python -m ml.models.pricing.train --cutoff 2025-12-01      # fixed benchmark
    python -m ml.models.pricing.train --cutoff 2025-12-01 --n-trials 100
    python -m ml.models.pricing.train --refresh                # force re-query from DB
"""
import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit

from ml.features.build_features import load_pricing_features

logger = logging.getLogger(__name__)

HOLDOUT_DAYS  = 90
N_TRIALS      = 50
QUANTILES     = [0.25, 0.50, 0.75]
ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "pricing"))

FEATURE_COLS = [
    "lead_source", "is_returning_customer",
    "prior_avg_job_value", "prior_job_count",
    "city_median_job_value", "city_rejection_rate",
    "month", "season",
]


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss for a single quantile."""
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)))


def _tune_hyperparams(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int) -> dict:
    """
    Run Optuna study to find best LightGBM hyperparams for P50 (median) quantile.
    Uses TimeSeriesSplit CV on training data. Returns best params dict.
    """
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "objective":        "quantile",
            "alpha":            0.50,
            "metric":           "quantile",
            "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 40),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "verbose":          -1,
            "random_state":     42,
        }
        fold_losses = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, categorical_feature="auto")
            preds = model.predict(X_val)
            fold_losses.append(_pinball_loss(y_val.to_numpy(), preds, 0.50))
        return float(np.mean(fold_losses))

    # Suppress Optuna's per-trial output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Optuna best P50 pinball loss: %.4f", study.best_value)
    return study.best_params


def train(cutoff: str | None = None, holdout_days: int = HOLDOUT_DAYS,
          n_trials: int = N_TRIALS, refresh: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_pricing_features(refresh=refresh)

    if cutoff:
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    else:
        cutoff_ts = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=holdout_days))

    train_df = df[df["estimate_date"] < cutoff_ts].copy()
    test_df  = df[df["estimate_date"] >= cutoff_ts].copy()
    logger.info("Cutoff: %s | Train: %d rows | Test: %d rows", cutoff_ts.date(), len(train_df), len(test_df))

    # Last 20% of training used for early stopping only
    val_idx  = int(len(train_df) * 0.8)
    tr       = train_df.iloc[:val_idx]
    val      = train_df.iloc[val_idx:]
    X_tr     = tr[FEATURE_COLS]
    X_val    = val[FEATURE_COLS]
    X_test   = test_df[FEATURE_COLS]
    y_tr     = tr["value"]
    y_val    = val["value"]
    y_test   = test_df["value"].to_numpy()

    # Naive baseline: always predict training median
    train_median = float(train_df["value"].median())
    logger.info("Training median (naive baseline): $%.0f", train_median)

    # Tune hyperparams on P50 using full training set (tr + val)
    logger.info("Running Optuna hyperparameter search (%d trials)...", n_trials)
    best_params = _tune_hyperparams(train_df[FEATURE_COLS], train_df["value"], n_trials)
    logger.info("Best params: %s", best_params)

    # Train 3 quantile models with the same best hyperparams
    models      = {}
    test_preds  = {}
    pinball_log = {}

    for q in QUANTILES:
        label = f"p{int(q*100)}"
        params = {
            **best_params,
            "objective":   "quantile",
            "alpha":        q,
            "metric":      "quantile",
            "verbose":     -1,
            "random_state": 42,
        }
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature="auto",
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        preds               = model.predict(X_test)
        loss                = _pinball_loss(y_test, preds, q)
        baseline_loss       = _pinball_loss(y_test, np.full_like(y_test, train_median), q)
        pct_improvement     = (baseline_loss - loss) / baseline_loss * 100

        logger.info(
            "%s  pinball=%.2f  baseline=%.2f  improvement=%.1f%%",
            label.upper(), loss, baseline_loss, pct_improvement,
        )
        models[label]     = model
        test_preds[label] = preds
        pinball_log[label] = {
            "pinball_loss":     round(loss, 4),
            "baseline_loss":    round(baseline_loss, 4),
            "pct_improvement":  round(pct_improvement, 2),
        }

    # Coverage: % of test values inside [p25, p75]
    inside   = np.sum((y_test >= test_preds["p25"]) & (y_test <= test_preds["p75"]))
    coverage = inside / len(y_test) * 100
    mean_width = float(np.mean(test_preds["p75"] - test_preds["p25"]))
    logger.info("Coverage: %.1f%%  |  Mean range width: $%.0f", coverage, mean_width)

    # Build training index for confidence scoring in predict.py
    training_index = train_df[["lead_source", "city_median_job_value"]].copy()
    training_index["lead_source"] = training_index["lead_source"].astype(str)
    training_index_records = training_index.to_dict(orient="records")

    # Training value distribution (stored in metadata for display)
    train_value_stats = {
        "min":    round(float(train_df["value"].min()), 2),
        "p25":    round(float(train_df["value"].quantile(0.25)), 2),
        "median": round(float(train_df["value"].median()), 2),
        "p75":    round(float(train_df["value"].quantile(0.75)), 2),
        "max":    round(float(train_df["value"].max()), 2),
    }

    mlflow.set_experiment("pricing_predictor")
    with mlflow.start_run(run_name=f"lgbm_quantile_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params({
            "train_cutoff":  str(cutoff_ts.date()),
            "holdout_days":  holdout_days if not cutoff else "n/a",
            "train_rows":    len(train_df),
            "test_rows":     len(test_df),
            "n_trials":      n_trials,
            "features":      ",".join(FEATURE_COLS),
            **{f"param_{k}": v for k, v in best_params.items()},
        })
        for label, stats in pinball_log.items():
            mlflow.log_metrics({f"{label}_{k}": v for k, v in stats.items()})
        mlflow.log_metrics({
            "coverage_pct":   round(coverage, 2),
            "mean_range_width": round(mean_width, 2),
        })

        for label, model in models.items():
            with open(os.path.join(ARTIFACTS_DIR, f"{label}.pkl"), "wb") as f:
                pickle.dump(model, f)

        metadata = {
            "feature_cols":      FEATURE_COLS,
            "quantiles":         QUANTILES,
            "best_params":       best_params,
            "train_cutoff":      str(cutoff_ts.date()),
            "trained_at":        datetime.now(timezone.utc).isoformat(),
            "train_median":      train_median,
            "train_value_stats": train_value_stats,
            "pinball_results":   pinball_log,
            "training_index":    training_index_records,
        }
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifacts(ARTIFACTS_DIR, artifact_path="models")
        logger.info("Artifacts saved to %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Train/test split date (YYYY-MM-DD). Defaults to today minus --holdout-days.")
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS,
                        help=f"Days to hold out as test set (default: {HOLDOUT_DAYS}).")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help=f"Number of Optuna trials for hyperparameter tuning (default: {N_TRIALS}).")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-query features from DB, bypassing the parquet cache.")
    args = parser.parse_args()
    train(cutoff=args.cutoff, holdout_days=args.holdout_days,
          n_trials=args.n_trials, refresh=args.refresh)
