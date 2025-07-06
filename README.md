
# VaporIQ Analytics Dashboard

A one‑file Streamlit app that explores the **VaporIQ synthetic survey** (≈2 000 rows, 44 columns) and
offers turnkey machine‑learning workflows:

| Tab | What you get |
|-----|--------------|
| **Data‑Viz** | Instant histograms / bar charts and a correlation heat‑map |
| **Classification** | KNN, Decision Tree, Random Forest, GBRT & ROC overlay |
| **Regression** | Linear, RF, GBRT – R²/MAE/RMSE + actual vs. predicted plot |
| **Clustering** | K‑means with elbow chart & interactive feature picker |
| **Association Rules** | Apriori mining with support / confidence / lift sliders |

Look‑and‑feel matches the previous `vaporiq_dashboard_v4` (grey‑on‑purple + watermark).

---

## Quick start (local)

```bash
git clone <your‑repo>
cd <your‑repo>
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
streamlit run app.py
```

Place **`vaporiq_synthetic_v2.csv`** in the same folder (or use the uploader in the sidebar).

---

## Deploy on Streamlit‑Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and `vaporiq_synthetic_v2.csv` to a new GitHub repo.  
2. Go to <https://share.streamlit.io>, sign in → **New app** → select the repo & branch.  
3. Set *Main file* to `app.py`, click **Deploy**.  
4. Done – tweaks auto‑re‑deploy on every `git push`.

---

© 2025 VaporIQ
