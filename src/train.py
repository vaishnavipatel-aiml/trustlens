import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, classification_report)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from features import build_features

SEED = 42


def evaluate(y_true, scores, k_fracs=(0.05, 0.10)):
    results = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    for k in k_fracs:
        n_top = max(1, int(len(scores) * k))
        top_idx = np.argsort(-scores)[:n_top]
        results[f"recall@{int(k*100)}%"] = float(y_true.iloc[top_idx].sum() / max(1, y_true.sum()))
        results[f"precision@{int(k*100)}%"] = float(y_true.iloc[top_idx].mean())
    return results


def main():
    raw = pd.read_csv("data/expenses.csv")
    df, feature_cols = build_features(raw)

    X = df[feature_cols].astype(float)
    y = raw["is_fraud"].values
    claims = raw.copy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, raw.index, test_size=0.2, random_state=SEED, stratify=y)

    # --- Baseline 1: Isolation Forest (unsupervised) ---
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)  # higher = more anomalous
    iso_results = evaluate(pd.Series(y_test, index=idx_test), pd.Series(iso_scores, index=idx_test))
    print("Isolation Forest:", json.dumps(iso_results, indent=2))

    # --- Main model: XGBoost (supervised) ---
    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=SEED,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    xgb_scores = xgb.predict_proba(X_test)[:, 1]
    xgb_results = evaluate(pd.Series(y_test, index=idx_test), pd.Series(xgb_scores, index=idx_test))
    print("XGBoost:", json.dumps(xgb_results, indent=2))

    # --- Save everything the dashboard needs ---
    claims_test = claims.loc[idx_test].copy()
    claims_test["risk_score"] = xgb_scores
    claims_test["iso_score"] = iso_scores

    joblib.dump(xgb, "data/xgb_model.pkl")
    X_test.to_parquet("data/X_test.parquet")
    claims_test.to_parquet("data/claims_scored.parquet")

    with open("data/metrics.json", "w") as f:
        json.dump({"isolation_forest": iso_results, "xgboost": xgb_results}, f, indent=2)

    print("\nSaved: xgb_model.pkl, X_test.parquet, claims_scored.parquet, metrics.json")
    print("\nClassification report @ top 10%:")
    n_top = int(len(xgb_scores) * 0.10)
    top_idx = np.argsort(-xgb_scores)[:n_top]
    y_pred = np.zeros_like(y_test)
    y_pred[top_idx] = 1
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
    