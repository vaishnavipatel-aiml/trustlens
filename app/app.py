 # --- FREE LOCAL LLM EXPLANATION ---
    st.markdown("### 🤖 AI Auditor Explanation (Local Model)")
    
    @st.cache_resource
    def load_local_llm():
        from transformers import pipeline
        # This is a small, free Google model that runs on CPU
        return pipeline("text2text-generation", model="google/flan-t5-base", max_length=60)

    if st.button("Generate Explanation"):
        with st.spinner("Local AI is analyzing the claim (this takes ~10 seconds)..."):
            generator = load_local_llm()
            
            # Extract top 3 features for the prompt
            top_feats = top.head(3)
            feat_text = "\n".join([f"- {r['feature']} = {r['value']:.2f} (Impact: {r['shap']:+.2f})" for _, r in top_feats.iterrows()])
            
            prompt = f"""Explain why this expense is unusual in one sentence. Do not use the word fraud.
            Amount: ${row['amount']:.2f}, Category: {row['category']}, Department: {row['department']}.
            Anomalies:
            {feat_text}"""
            
            try:
                # Generate the text locally
                result = generator(prompt)[0]['generated_text']
                st.success(result)
            except Exception as e:
                st.error(f"Error: {e}")
    # ---------------------------
