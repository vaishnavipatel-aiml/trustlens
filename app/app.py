import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="wide")

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

st.title("🔍 TrustLens — Expense Claim Triage")
st.caption("Flags **unusual** claims for human review. Does not determine fraud.")

tab_queue, tab_detail, tab_card = st.tabs(["📋 Review Queue", "🔎 Claim Detail", "📊 Model Card"])

# ---------- TAB 1 ----------
with tab_queue:
    st.markdown("### Prioritized review queue")
    col1, col2, col3 = st.columns(3)
    dept_filter = col1.multiselect("Department", sorted(claims["department"].unique()))
    min_score = col2.slider("Min risk score", 0.0, 1.0, 0.5, 0.01)
    top_k = col3.number_input("Show top K", 10, 500, 50, 10)

    view = claims.copy()
    if dept_filter:
        view = view[view["department"].isin(dept_filter)]
    view = view[view["risk_score"] >= min_score].sort_values("risk_score", ascending=False).head(top_k)

    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged claims", len(view))
    m2.metric("Avg risk score", f"{view['risk_score'].mean():.3f}" if len(view) else "—")
    m3.metric("Avg amount", f"${view['amount'].mean():,.0f}" if len(view) else "—")

    st.dataframe(
        view[["claim_id", "employee_id", "department", "category",
              "amount", "submit_datetime", "vendor", "risk_score"]],
        use_container_width=True, hide_index=True,
    )
    st.download_button("⬇️ Download flagged claims (CSV)",
                       view.to_csv(index=False), "flagged_claims.csv")

# ---------- TAB 2 ----------
with tab_detail:
    st.markdown("### Inspect a specific claim")
    top_claims = claims.sort_values("risk_score", ascending=False).head(300)
    cid = st.selectbox("Select claim", top_claims["claim_id"].tolist())

    row_pos = claims.index[claims["claim_id"] == cid][0]
    row = claims.iloc[row_pos]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amount", f"${row['amount']:,.2f}")
    c2.metric("Risk score", f"{row['risk_score']:.3f}")
    c3.metric("Department", row["department"])
    c4.metric("Role", row["role_level"])

    st.write(f"**Vendor:** {row['vendor']}  |  **Category:** {row['category']}")
    st.write(f"**Submitted:** {str(row['submit_datetime'])[:16]}  |  **Memo:** _{row['memo']}_")

    st.markdown("#### Why is this claim unusual?")

    # SHAP waterfall-ish
    sv = explainer.shap_values(X_test.iloc[[row_pos]])[0]
    contrib = pd.DataFrame({
        "feature": X_test.columns,
        "shap": sv,
        "value": X_test.iloc[row_pos].values,
    })
    contrib["abs"] = contrib["shap"].abs()
    top = contrib.sort_values("abs", ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top["shap"]]
    ax.barh(top["feature"][::-1], top["shap"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP contribution (red = pushes toward anomaly)")
    st.pyplot(fig)

            # --- NEW LLM EXPLANATION (IMPROVED) ---
    st.markdown("### 🤖 AI Auditor Explanation")
    
    # Check if the key is in Streamlit Secrets (for the public website)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        # If not, ask for it on the screen
        api_key = st.text_input("Paste your OpenAI API Key here (it starts with sk-):", type="password")

    if st.button("Generate Explanation"):
        if not api_key:
            st.warning("Please paste your API key first!")
        else:
            with st.spinner("AI is analyzing the claim..."):
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                # Extract top 3 features for the prompt
                top_feats = top.head(3)
                feat_text = "\n".join([f"- {r['feature']} = {r['value']:.2f} (Impact: {r['shap']:+.2f})" for _, r in top_feats.iterrows()])
                
                prompt = f"""You are an audit assistant. Write ONE concise sentence (max 30 words) explaining why this expense claim is unusual.
Rules: Never use the word 'fraud'. Use 'unusual' or 'warrants review'. Quantify deviations.
Claim: Amount ${row['amount']:.2f}, {row['category']}, Dept: {row['department']}, Role: {row['role_level']}, Vendor: {row['vendor']}.
Top Anomalies:
{feat_text}
Output exactly one sentence:"""

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60,
                        temperature=0.3
                    )
                    st.success(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error: {e}")
    # ---------------------------

    b1, b2 = st.columns(2)
    if b1.button("✅ Mark reviewed — no action"):
        st.success("Logged.")
    if b2.button("🚩 Escalate to audit"):
        st.warning("Escalated.")

# ---------- TAB 3 ----------
with tab_card:
    st.markdown("""
### Intended use
Assist auditors in prioritizing expense claims for human review.

### Not intended for
- Automated accusation or denial
- Decisions without human review

### Data
20,000 synthetic claims with 5 documented anomaly patterns injected at 3%.

### Metrics (holdout, 20%)
PR-AUC and Recall@top-5% from `src/train.py`.

### Known limitations
- Synthetic data may not reflect real corporate spending.
- False positive rate varies by department (see chart below).
- Not adversarially robust.

### Ethical notes
- Flagged ≠ fraudulent.
- Employees should be able to contest flags.
- Per-department FPR should be monitored.
""")

    st.markdown("#### Fairness: flag rate by department (among non-fraud claims)")
    thr = 0.5
    claims["flagged"] = claims["risk_score"] >= thr
    nonfraud = claims[claims["is_fraud"] == 0]
    fpr = nonfraud.groupby("department")["flagged"].mean().sort_values(ascending=False)
    st.bar_chart(fpr)
    st.caption(f"Overall false-positive rate: {nonfraud['flagged'].mean():.3f}")



