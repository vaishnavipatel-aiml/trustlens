import pandas as pd
import numpy as np
import joblib
import shap
from features import build_features


def rule_based_explanation(row, shap_row, feature_names):
    """Convert top SHAP drivers into a human sentence. No LLM needed."""
    order = np.argsort(-np.abs(shap_row))[:3]
    reasons = []

    for i in order:
        f = feature_names[i]
        v = row[f] if f in row else None
        if f == "amount_zscore" and v is not None:
            reasons.append(f"amount is {abs(v):.1f}× the peer median for this role and category")
        elif f == "amount_zscore_dept" and v is not None:
            reasons.append(f"amount deviates {abs(v):.1f}σ from the department average")
        elif f == "submit_hour" and v is not None:
            reasons.append(f"submitted at {int(v)}:00, outside typical working hours")
        elif f == "is_weekend" and v == 1:
            reasons.append("submitted on a weekend")
        elif f == "is_late_night" and v == 1:
            reasons.append("submitted late at night")
        elif f == "vendor_approved" and v == 0:
            reasons.append("vendor is not on the approved list")
        elif f == "is_round_number" and v == 1:
            reasons.append("amount is a suspiciously round number")
        elif f == "days_to_submit" and v is not None and v > 30:
            reasons.append(f"submitted {int(v)} days after the expense occurred")
        elif f == "memo_is_vague" and v == 1:
            reasons.append("memo is unusually vague")

    if not reasons:
        reasons.append("multiple subtle deviations from the peer baseline")

    return "This claim is unusual because the " + "; ".join(reasons[:2]) + "."


def generate_explanations(top_n=50):
    model = joblib.load("data/xgb_model.pkl")
    X_test = pd.read_parquet("data/X_test.parquet")
    claims = pd.read_parquet("data/claims_scored.parquet")
    feature_names = list(X_test.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Re-attach features to claims rows in same order
    X_test_reset = X_test.reset_index(drop=True)
    claims_reset = claims.reset_index(drop=True)

    # Generate explanations for top-N riskiest claims
    top_idx = np.argsort(-claims_reset["risk_score"].values)[:top_n]
    explanations = []
    for i in top_idx:
        row = X_test_reset.iloc[i]
        sv = shap_values[i]
        text = rule_based_explanation(row, sv, feature_names)
        explanations.append({
            "row_index": int(i),
            "claim_id": claims_reset.iloc[i]["claim_id"],
            "explanation": text,
            "top_features": [feature_names[j] for j in np.argsort(-np.abs(sv))[:3]],
        })

    pd.DataFrame(explanations).to_parquet("data/explanations.parquet", index=False)
    print(f"Generated {len(explanations)} explanations → data/explanations.parquet")
    print("\nSample:")
    print(explanations[0]["explanation"])


if __name__ == "__main__":
    generate_explanations()
    