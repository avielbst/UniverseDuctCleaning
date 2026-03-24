"""
Evaluate the trained pricing predictor and print a full performance report.

Metrics: pinball loss (P25/P50/P75) vs naive median baseline, coverage,
mean range width, MAE on P50, feature importance.

Usage:
    python -m ml.models.pricing.evaluate
"""
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml.features.build_features import load_pricing_features

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "pricing"))


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)))


def _load_artifacts():
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata found at {meta_path}. Run train.py first.")
    with open(meta_path) as f:
        meta = json.load(f)
    models = {}
    for q in meta["quantiles"]:
        label = f"p{int(q*100)}"
        with open(os.path.join(ARTIFACTS_DIR, f"{label}.pkl"), "rb") as f:
            models[label] = pickle.load(f)
    return models, meta


def evaluate():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models, meta = _load_artifacts()
    feature_cols  = meta["feature_cols"]
    cutoff_ts     = pd.Timestamp(meta["train_cutoff"], tz="UTC")
    train_median  = meta["train_median"]

    df       = load_pricing_features()
    train_df = df[df["estimate_date"] < cutoff_ts].copy()
    test_df  = df[df["estimate_date"] >= cutoff_ts].copy()
    y_test   = test_df["value"].to_numpy()

    logger.info("Cutoff: %s | Train: %d | Test: %d", cutoff_ts.date(), len(train_df), len(test_df))

    preds = {}
    for label, model in models.items():
        preds[label] = model.predict(test_df[feature_cols])

    # ── Print report ──────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  PRICING PREDICTOR - EVALUATION REPORT")
    print(f"  Trained: {meta['trained_at'][:10]}  |  Cutoff: {meta['train_cutoff']}")
    print(f"  Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print(f"  Training median (naive baseline): ${train_median:,.0f}")
    print(f"{'='*70}\n")

    # Pinball loss per quantile
    print(f"QUANTILE PERFORMANCE")
    print(f"{'Quantile':<10} {'Pinball':>8} {'Baseline':>8} {'Improvement':>12}")
    print("-" * 42)
    quantile_results = []
    for q in meta["quantiles"]:
        label    = f"p{int(q*100)}"
        loss     = _pinball_loss(y_test, preds[label], q)
        baseline = _pinball_loss(y_test, np.full_like(y_test, train_median), q)
        pct_imp  = (baseline - loss) / baseline * 100
        print(f"{label.upper():<10} {loss:>8.2f} {baseline:>8.2f} {pct_imp:>11.1f}%")
        quantile_results.append({"quantile": q, "label": label, "pinball": round(loss, 4),
                                  "baseline": round(baseline, 4), "pct_improvement": round(pct_imp, 2)})

    # P50 MAE
    mae_model    = mean_absolute_error(y_test, preds["p50"])
    mae_baseline = mean_absolute_error(y_test, np.full_like(y_test, train_median))
    print(f"\nP50 MAE: ${mae_model:,.0f}  |  Baseline MAE: ${mae_baseline:,.0f}  "
          f"|  Improvement: {(mae_baseline - mae_model) / mae_baseline * 100:.1f}%")

    # Coverage and range width
    inside     = np.sum((y_test >= preds["p25"]) & (y_test <= preds["p75"]))
    coverage   = inside / len(y_test) * 100
    mean_width = float(np.mean(preds["p75"] - preds["p25"]))
    print(f"\nCOVERAGE & RANGE")
    print(f"  [P25, P75] coverage : {coverage:.1f}%  (target ~50%)")
    print(f"  Mean range width    : ${mean_width:,.0f}")

    # Predicted range distribution
    print(f"\nPREDICTED RANGE DISTRIBUTION (test set)")
    print(f"  {'':15} {'P25':>8} {'P50':>8} {'P75':>8} {'Width':>8}")
    print(f"  {'Min':15} ${np.min(preds['p25']):>7,.0f} ${np.min(preds['p50']):>7,.0f} "
          f"${np.min(preds['p75']):>7,.0f} ${np.min(preds['p75']-preds['p25']):>7,.0f}")
    print(f"  {'Median':15} ${np.median(preds['p25']):>7,.0f} ${np.median(preds['p50']):>7,.0f} "
          f"${np.median(preds['p75']):>7,.0f} ${np.median(preds['p75']-preds['p25']):>7,.0f}")
    print(f"  {'Max':15} ${np.max(preds['p25']):>7,.0f} ${np.max(preds['p50']):>7,.0f} "
          f"${np.max(preds['p75']):>7,.0f} ${np.max(preds['p75']-preds['p25']):>7,.0f}")

    # Actual value distribution (context)
    print(f"\nACTUAL VALUE DISTRIBUTION (test set)")
    for pct, label in [(25, "P25"), (50, "Median"), (75, "P75")]:
        print(f"  {label:10}: ${np.percentile(y_test, pct):,.0f}")

    # Feature importance from P50
    model_p50    = models["p50"]
    imp          = model_p50.feature_importances_
    imp_pct      = imp / imp.sum() * 100
    feat_imp     = sorted(zip(model_p50.feature_name_, imp_pct), key=lambda x: -x[1])
    print(f"\nFEATURE IMPORTANCE (P50 model)")
    print("-" * 40)
    for feat, pct in feat_imp:
        bar = "#" * int(pct / 2)
        print(f"  {feat:<35} {pct:>5.1f}%  {bar}")

    # Best hyperparams
    print(f"\nBEST HYPERPARAMS (Optuna, {meta.get('pinball_results', {}).get('p50', {}).get('pinball_loss', 'n/a')} P50 pinball)")
    for k, v in meta["best_params"].items():
        print(f"  {k:<25}: {v}")

    # ── Save report ───────────────────────────────────────────────────────────

    report = {
        "train_cutoff":    meta["train_cutoff"],
        "evaluated_at":    pd.Timestamp.now(tz="UTC").isoformat(),
        "train_rows":      len(train_df),
        "test_rows":       len(test_df),
        "train_median":    train_median,
        "quantile_results": quantile_results,
        "p50_mae":         round(mae_model, 2),
        "p50_mae_baseline": round(mae_baseline, 2),
        "coverage_pct":    round(coverage, 2),
        "mean_range_width": round(mean_width, 2),
        "feature_importance": [{"feature": f, "importance_pct": round(i, 2)} for f, i in feat_imp],
        "best_params":     meta["best_params"],
    }
    out_path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}\n")


if __name__ == "__main__":
    evaluate()
