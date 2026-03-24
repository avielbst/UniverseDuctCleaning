"""
Evaluate the trained upsell classifiers and print a full performance report.

Loads models and metadata from ml/artifacts/upsell/, rebuilds the feature
matrix, and evaluates on the held-out test set defined by the stored cutoff.

Usage:
    python -m ml.models.upsell.evaluate
"""
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)

from ml.features.build_features import build_upsell_features

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "upsell"))


def _load_artifacts():
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata found at {meta_path}. Run train.py first.")
    with open(meta_path) as f:
        meta = json.load(f)

    models = {}
    for label in meta["labels"]:
        path = os.path.join(ARTIFACTS_DIR, f"{label.replace(' ', '_')}.pkl")
        with open(path, "rb") as f:
            models[label] = pickle.load(f)
    return models, meta


def _baseline_probs(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str) -> np.ndarray:
    """P(label=1 | first_service) from training data. Fallback to overall prevalence."""
    fallback = train_df[label].mean()
    rates = train_df.groupby("first_service", observed=True)[label].mean().to_dict()
    return test_df["first_service"].map(rates).fillna(fallback).to_numpy()


def _optimal_f1_threshold(y_true, y_prob):
    """Return (threshold, precision, recall, f1) at the point maximising F1."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    idx = np.argmax(f1)
    # precision_recall_curve appends a final point with no matching threshold
    thr = thresholds[idx] if idx < len(thresholds) else 1.0
    return float(thr), float(prec[idx]), float(rec[idx]), float(f1[idx])


def _confidence_tier(n_train_pos: int) -> str:
    """Map training positives to a confidence tier for the predict function."""
    if n_train_pos >= 100:
        return "high"
    if n_train_pos >= 40:
        return "medium"
    return "low"


def evaluate():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models, meta = _load_artifacts()
    feature_cols = meta["feature_cols"]
    cutoff_ts    = pd.Timestamp(meta["train_cutoff"], tz="UTC")
    label_stats  = meta.get("label_stats", {})

    df       = build_upsell_features()
    train_df = df[df["job_date"] < cutoff_ts].copy()
    test_df  = df[df["job_date"] >= cutoff_ts].copy()

    logger.info("Cutoff: %s | Train: %d | Test: %d", cutoff_ts.date(), len(train_df), len(test_df))

    rows = []
    all_importances  = []  # collect per-label feature importances for aggregate view
    importance_names = []  # feature names from first lgbm model (all share same feature set)

    for label in meta["labels"]:
        model       = models[label]
        stats       = label_stats.get(label, {})
        base_rates  = stats.get("baseline_rates", {})
        base_fallback = stats.get("baseline_fallback", train_df[label].mean())

        y_true = test_df[label].to_numpy()
        b_prob = _baseline_probs(train_df, test_df, label)

        if stats.get("use_baseline", False):
            y_prob = b_prob
        else:
            # Add per-label baseline_prob feature — must match training setup
            X_test_l = test_df[feature_cols].assign(
                baseline_prob=test_df["first_service"].map(base_rates).fillna(base_fallback)
            )
            y_prob = model.predict_proba(X_test_l)[:, 1]

        n_pos_test = int(y_true.sum())
        prevalence = y_true.mean() * 100

        # Skip labels with no positive examples in test — metrics are undefined
        if n_pos_test == 0:
            logger.warning("Skipping %s — no positives in test set", label)
            continue

        auroc    = roc_auc_score(y_true, y_prob)
        avg_prec = average_precision_score(y_true, y_prob)
        brier    = brier_score_loss(y_true, y_prob)

        b_auroc  = roc_auc_score(y_true, b_prob) if len(np.unique(y_true)) > 1 else float("nan")

        opt_thr, opt_prec, opt_rec, opt_f1 = _optimal_f1_threshold(y_true, y_prob)

        n_train_pos = label_stats.get(label, {}).get("n_train_pos", "?")
        confidence  = _confidence_tier(n_train_pos) if isinstance(n_train_pos, int) else "?"

        # Precision at the stored precision-optimized threshold
        prec_thr    = label_stats.get(label, {}).get("precision_threshold", 0.5)
        prec_at_thr = precision_score(y_true, (y_prob >= prec_thr).astype(int), zero_division=0.0)
        delta       = auroc - b_auroc if not np.isnan(b_auroc) else float("nan")
        use_baseline = label_stats.get(label, {}).get("use_baseline", False)

        if not stats.get("use_baseline", False):
            all_importances.append(model.feature_importances_)
            if not importance_names:
                importance_names = list(model.feature_name_)

        rows.append({
            "label":          label,
            "n_test_pos":     n_pos_test,
            "prevalence":     round(prevalence, 1),
            "n_train_pos":    n_train_pos,
            "confidence":     confidence,
            "auroc":          round(auroc, 4),
            "avg_precision":  round(avg_prec, 4),
            "brier":          round(brier, 4),
            "baseline_auroc": round(b_auroc, 4) if not np.isnan(b_auroc) else None,
            "beats_baseline": auroc > b_auroc if not np.isnan(b_auroc) else None,
            "delta":          round(delta, 4) if not np.isnan(delta) else None,
            "prec_threshold": round(prec_thr, 3),
            "prec_at_thr":    round(prec_at_thr, 3),
            "use_baseline":   use_baseline,
            "opt_threshold":  round(opt_thr, 3),
            "opt_precision":  round(opt_prec, 3),
            "opt_recall":     round(opt_rec, 3),
            "opt_f1":         round(opt_f1, 3),
        })

    # ── Print report ──────────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  UPSELL CLASSIFIER - EVALUATION REPORT")
    print(f"  Trained: {meta['trained_at'][:10]}  |  Cutoff: {meta['train_cutoff']}")
    print(f"  Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print(f"{'='*80}\n")

    # Per-label metrics
    print(f"{'LABEL':<42} {'SUP':>4} {'PREV':>5} {'AUROC':>6} {'AvgP':>6} {'BRIER':>6} {'BASE':>6} {'DELTA':>6} {'P@thr':>6} {'USE':>5} {'CONF':>6}")
    print("-" * 115)
    for r in rows:
        base  = f"{r['baseline_auroc']:.4f}" if r["baseline_auroc"] is not None else "   n/a"
        delta = f"{r['delta']:+.4f}"         if r["delta"] is not None           else "   n/a"
        src   = "base" if r["use_baseline"] else "lgbm"
        print(
            f"{r['label']:<42} {r['n_test_pos']:>4} {r['prevalence']:>4.1f}%"
            f" {r['auroc']:>6.4f} {r['avg_precision']:>6.4f} {r['brier']:>6.4f}"
            f" {base:>6} {delta:>6} {r['prec_at_thr']:>6.3f} {src:>5} {r['confidence']:>6}"
        )

    # Summary
    aurocs      = [r["auroc"] for r in rows]
    beats_count = sum(1 for r in rows if r["beats_baseline"])
    above_065   = sum(1 for a in aurocs if a > 0.65)
    print(f"\nSUMMARY")
    print(f"  Mean AUROC      : {np.mean(aurocs):.4f}")
    print(f"  Labels > 0.65   : {above_065}/{len(rows)}")
    print(f"  Beats baseline  : {beats_count}/{len(rows)}")

    # Optimal thresholds — needed by predict function
    print(f"\nOPTIMAL THRESHOLDS (max-F1)")
    print(f"{'LABEL':<42} {'THR':>5} {'PREC':>6} {'REC':>6} {'F1':>6}")
    print("-" * 65)
    for r in rows:
        print(f"{r['label']:<42} {r['opt_threshold']:>5.3f} {r['opt_precision']:>6.3f} {r['opt_recall']:>6.3f} {r['opt_f1']:>6.3f}")

    # Feature importance
    if all_importances:
        imp_matrix = np.vstack(all_importances)
        mean_imp   = imp_matrix.mean(axis=0)
        mean_imp   = mean_imp / mean_imp.sum() * 100
        feat_imp   = sorted(zip(importance_names, mean_imp), key=lambda x: -x[1])

        print(f"\nFEATURE IMPORTANCE (mean across all labels)")
        print("-" * 40)
        for feat, imp in feat_imp:
            bar = "#" * int(imp / 2)
            print(f"  {feat:<35} {imp:>5.1f}%  {bar}")

    # ── Save report ───────────────────────────────────────────────────────────

    report = {
        "train_cutoff":    meta["train_cutoff"],
        "evaluated_at":    pd.Timestamp.now(tz="UTC").isoformat(),
        "train_rows":      len(train_df),
        "test_rows":       len(test_df),
        "summary": {
            "mean_auroc":     round(np.mean(aurocs), 4),
            "labels_above_065": above_065,
            "labels_total":   len(rows),
            "beats_baseline": beats_count,
        },
        "labels": rows,
        "feature_importance": [
            {"feature": f, "importance_pct": round(i, 2)} for f, i in feat_imp
        ] if all_importances else [],
    }

    out_path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}\n")


if __name__ == "__main__":
    evaluate()
