# VaporIQ Streamlit Dashboard (Minimal)

Self‑contained Streamlit app showcasing:

- **Classification** (KNN, Decision Tree, Random Forest, GBRT)  
- **Clustering** (K‑Means with k‑slider)  
- **Association‑Rule Mining** (Apriori, top‑10 rules)  
- **Regression** (Linear, Ridge, Lasso, Decision Tree with hyper‑tuning)  
- **Data Visualisation** (10 example insights)

Everything lives in **`app.py`**—no extra modules required.

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push these three files (`app.py`, `requirements.txt`, `README.md`) to a new GitHub repo.  
2. On **streamlit.io/cloud** → *New app* → choose `app.py`.  
3. Provide the RAW URL or local path to your CSV in the sidebar when the app loads.