import pandas as pd
import numpy as np

APPROVED_VENDORS = {
    "Chipotle", "Local Bistro", "Starbucks", "Delta SkyClub",
    "Delta", "United", "Marriott", "Hertz",
    "AWS", "GitHub", "Slack", "Notion", "Figma",
    "Staples", "Amazon Business", "Uline",
    "Topgolf", "Eventbrite", "AMC",
    "Coursera", "O'Reilly", "Udemy Business",
}


def build_features(df):
    df = df.copy()
    df["submit_datetime"] = pd.to_datetime(df["submit_datetime"])
    df["expense_date"] = pd.to_datetime(df["expense_date"])

    df["submit_hour"] = df["submit_datetime"].dt.hour
    df["submit_dow"] = df["submit_datetime"].dt.dayofweek
    df["is_weekend"] = (df["submit_dow"] >= 5).astype(int)
    df["is_late_night"] = ((df["submit_hour"] >= 22) | (df["submit_hour"] <= 5)).astype(int)
    df["days_to_submit"] = (df["submit_datetime"] - df["expense_date"]).dt.days.clip(lower=0)

    # Peer-relative amount: z-score within category × role
    grp = df.groupby(["category", "role_level"])["amount"]
    df["amount_zscore"] = (df["amount"] - grp.transform("mean")) / (grp.transform("std") + 1e-6)

    # Department-level amount z-score too
    dept_grp = df.groupby("department")["amount"]
    df["amount_zscore_dept"] = (df["amount"] - dept_grp.transform("mean")) / (dept_grp.transform("std") + 1e-6)

    df["vendor_approved"] = df["vendor"].isin(APPROVED_VENDORS).astype(int)
    df["memo_length"] = df["memo"].str.len()
    df["memo_is_vague"] = df["memo"].str.lower().str.contains("misc|other|partial").astype(int)
    df["is_round_number"] = (df["amount"] % 100 == 0).astype(int)

    # One-hot encode categoricals (drop is_fraud/fraud_type/ids)
    cat_cols = ["department", "role_level", "category"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    feature_cols = [c for c in df.columns
                    if c not in ["claim_id", "employee_id", "vendor", "memo",
                                 "submit_datetime", "expense_date", "is_fraud", "fraud_type"]]
    return df, feature_cols


if __name__ == "__main__":
    df = pd.read_csv("data/expenses.csv")
    df_feat, cols = build_features(df)
    df_feat.to_parquet("data/features.parquet", index=False)
    print(f"Built {len(cols)} features for {len(df_feat)} rows")
    print("Features:", cols[:10], "...")
    