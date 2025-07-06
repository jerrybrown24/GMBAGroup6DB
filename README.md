# VaporIQ Streamlit Dashboard

**Purpose**  
A one‑stop Streamlit app that demonstrates VaporIQ’s five core USPs plus a bonus “Insights” tab:

1. Personal Pricing (regression)
2. Segment Explorer (clustering)
3. Flavor & Mood Pairing (association + classification)
4. Churn & Referral Radar (classification)
5. Data‑Trust Console (privacy tiers)
6. Statistical Insights (top‑10 visual nuggets)

## Quick start
```bash
git clone <your‑fork‑url>
cd vaporiq_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The uploaded **`data/vaporiq_data.csv`** is synthetic but mirrors VaporIQ survey fields.

**Dev notes**
* Each tab function is isolated in *app.py*—extend or replace the baseline models.
* Heavy models are trained on the fly for demo; cache with `@st.cache_resource` for production.
* No secrets needed; app runs entirely client‑side.
