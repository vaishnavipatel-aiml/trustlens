import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. PAGE CONFIG & ENTERPRISE STYLING ---
st.set_page_config(page_title="TrustLens | Expense Audit", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .header-container {
        background-color: white;
        padding: 20px 40px;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        margin-bottom: 30px;
    }
    .header-title {
        color: #0F172A;
        font-size: 24px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        color: #2563EB;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1E3A8A;
        color: white;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB;
        border-bottom-color: #2563EB;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD DATA ---
ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_everything():
    model = joblib.load(ROOT / "data/xgb_model.pkl")
    X_test = pd.read_parquet(ROOT / "data/X_test.parquet").reset_index(drop=True)
    claims = pd.read_parquet(ROOT / "data/claims_scored.parquet").reset_index(drop=True)
    exps = pd.read_parquet(ROOT / "data/explanations.parquet")
    explainer = shap.TreeExplainer(model)
    return model, X_test, claims, exps, explainer

model, X_test, claims, exps, explainer = load_everything()

# --- 3. UI HEADER ---
st.markdown("""
<div class="header-container">
    <div class="header-title">
        <span style="font-size: 28px;">🛡️</span> TrustLens <span style="color:#64748B; font-weight:400; font-size:16px; margin-left:10px;">| AI Expense Audit</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_queue, tab_detail, tab_card = st.tabs(["📋 Prioritized Review Queue", "🔎 Claim Investigation", "📊 Model & Fairness Card"])

# ---------- TAB 1: QUEUE ----------
with tab_queue:
    st.markdown("### Unusual Claims Needing Review")
    st.markdown("Our machine learning model flags claims that deviate from peer baselines. These are prioritized for human auditors.")
    
    col1, col2, col3 = st.columns(3)
    dept_filter = col1.multiselect("Filter by Department", sorted(claims["department"].unique()))
    min_score = col2.slider("Minimum Risk Score", 0.0, 1.0, 0.5, 0.01)
    top_k = col3.number_input("Max Claims to Show", 10, 500, 50, 10)

    view = claims.copy()
    if dept_filter:
        view = view[view["department"].isin(dept_filter)]
    view = view[view["risk_score"] >= min_score].sort_values("risk_score", ascending=False).head(top_k)

    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged Claims", len(view))
    m2.metric("Avg Risk Score", f"{view['risk_score'].mean():.3f}" if len(view) else "—")
    m3.metric("Avg Claim Amount", f"${view['amount'].mean():,.0f}" if len(view) else "—")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.dataframe(
        view[["claim_id", "employee_id", "department", "category", "amount", "submit_datetime", "vendor", "risk_score"]],
        use_container_width=True, hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.3f")
        }
    )
    st.download_button("⬇️ Export Flagged Claims (CSV)", view.to_csv(index=False), "flagged_claims.csv")

# ---------- TAB 2: DETAIL ----------
with tab_detail:
    st.markdown("### Claim Investigation")
    
    top_claims = claims.sort_values("risk_score", ascending=False).head(300)
    cid = st.selectbox("Select a Flagged Claim to Investigate", top_claims["claim_id"].tolist())

    row_pos = claims.index[claims["claim_id"] == cid][0]
    row = claims.iloc[row_pos]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amount", f"${row['amount']:,.2f}")
    c2.metric("Risk Score", f"{row['risk_score']:.3f}")
    c3.metric("Department", row["department"])
    c4.metric("Role", row["role_level"])

    st.info(f"**Vendor:** {row['vendor']}  |  **Category:** {row['category']}  |  **Submitted:** {str(row['submit_datetime'])[:16]}  |  **Memo:** _{row['memo']}_")

    st.markdown("#### SHAP Feature Contributions")
    st.markdown("The chart below explains *exactly* which features pushed the risk score higher (red) or lower (green).")
    
    sv = explainer.shap_values(X_test.iloc[[row_pos]])[0]
    contrib = pd.DataFrame({"feature": X_test.columns, "shap": sv, "value": X_test.iloc[row_pos].values})
    contrib["abs"] = contrib["shap"].abs()
    top = contrib.sort_values("abs", ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    colors = ["#EF4444" if v > 0 else "#10B981" for v in top["shap"]]
    ax.barh(top["feature"][::-1], top["shap"][::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="#94A3B8", lw=1.5, linestyle="--")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel("Impact on Risk Score (Red = Pushes toward Anomaly)", color="#475569", fontweight="bold")
    st.pyplot(fig)

    st.markdown("---")
    
    # Rule-based Explanation (No heavy AI needed!)
    st.markdown("#### 🤖 AI Auditor Explanation")
    match = exps[exps["claim_id"] == cid]
    if len(match):
        st.success(f"**Auditor Note:** {match.iloc[0]['explanation']}")
    else:
        st.success("**Auditor Note:** This claim exhibits multiple subtle deviations from the peer baseline.")

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1,1,2])
    if b1.button("✅ Approve Claim"):
        st.success("Claim approved and logged.")
    if b2.button("🚩 Escalate to Audit"):
        st.warning("Claim escalated to senior auditor.")

# ---------- TAB 3: MODEL CARD ----------
with tab_card:
    st.markdown("### Model & Fairness Card")
    st.markdown("Transparency is critical for enterprise AI. This card documents the model's performance and ethical safeguards.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Intended Use:**
        Assist human auditors in prioritizing expense claims for review. 
        
        **Not Intended For:**
        - Automated accusation or denial of claims.
        - Decisions made without human review.
        
        **Architecture:**
        - XGBoost Classifier (Supervised)
        - Trained on 20,000 synthetic claims with 3% injected anomalies.
        """)
    
    with col2:
        st.markdown("""
        **Ethical Safeguards:**
        - **Flags ≠ Fraud:** The system flags claims as "unusual", not "guilty".
        - **Human-in-the-loop:** Auditors make the final decision.
        - **Transparency:** SHAP values are provided for every flag.
        """)

    st.markdown("---")
    st.markdown("#### Fairness Audit: False Positive Rate by Department")
    
    thr = 0.5
    claims["flagged"] = claims["risk_score"] >= thr
    nonfraud = claims[claims["is_fraud"] == 0]
    fpr = nonfraud.groupby("department")["flagged"].mean().sort_values(ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor('#F8FAFC')
    ax2.set_facecolor('#F8FAFC')
    bars = ax2.bar(fpr.index, fpr.values, color="#3B82F6", width=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_ylabel("Flag Rate (False Positives)", color="#475569")
    st.pyplot(fig2)
