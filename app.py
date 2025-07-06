# app.py ────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import pathlib

# ───────────────────────────────────────────────────────────────────
# CONFIG & DATA LOADING
# ───────────────────────────────────────────────────────────────────
DATA_PATH = pathlib.Path(__file__).with_name("vaporiq_data.csv")

NUMERIC_COLS_TO_CLEAN = [
    "willingness_to_pay_usd",
    "monthly_spend_usd",
    "weekly_consumption_ml",
    "nicotine_strength_mgml",
    "age",
]

def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters and convert to float."""
    cleaned = (
        series.astype(str)
              .str.replace(r"[^0-9.\-]", "", regex=True)   # drop $, commas, text
    )
    return pd.to_numeric(cleaned, errors="coerce")

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load CSV from disk or user upload; return a cleaned DataFrame."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        st.warning(
            f"⚠️  Couldn’t find **{DATA_PATH.name}** next to *app.py*."
            " Upload it manually and I’ll use that copy."
        )
        uploaded = st.file_uploader("Upload vaporiq_data.csv", type="csv")
        if uploaded is None:
            st.stop()
        df = pd.read_csv(uploaded)

    df.columns = df.columns.str.strip()          # remove stray spaces

    # Clean numeric-as-text columns
    for col in NUMERIC_COLS_TO_CLEAN:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    return df

df = load_data()

# ───────────────────────────────────────────────────────────────────
# STREAMLIT LAYOUT
# ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VaporIQ Dashboard", layout="wide")
st.title("🚀 VaporIQ – Hyper-Personalized Vape Subscription Intelligence")

tabs = st.tabs([
    "Personal Pricing",
    "Segment Explorer",
    "Flavor & Mood",
    "Churn & Referral",
    "Data-Trust Console",
    "Statistical Insights"
])

# ==================================================================
# 1 · PERSONAL PRICING  (Regression)
# ==================================================================
with tabs[0]:
    st.header("💸 Personal Pricing")

    target = "willingness_to_pay_usd"
    num_cols = ["age", "weekly_consumption_ml",
                "nicotine_strength_mgml", "monthly_spend_usd"]
    cat_cols = ["country", "device_type", "primary_vape_motivation"]

    df_mod = df.dropna(subset=[target])
    X, y = df_mod[num_cols + cat_cols], df_mod[target]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    model = Pipeline([("pre", pre),
                      ("reg", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))])
    model.fit(X, y)

    st.subheader("Predict a user’s price ceiling")
    sample = {}
    for c in num_cols:
        sample[c] = st.number_input(
            c, float(df[c].min()), float(df[c].max()), float(df[c].median()))
    for c in cat_cols:
        sample[c] = st.selectbox(c, sorted(df[c].dropna().unique()))

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([sample]))[0]
        st.metric("Estimated willingness to pay (USD)", f"{pred:,.2f}")

# ==================================================================
# 2 · SEGMENT EXPLORER  (Clustering)
# ==================================================================
with tabs[1]:
    st.header("🔍 Segment Explorer")

    seg_cols = ["age", "weekly_consumption_ml",
                "monthly_spend_usd", "willingness_to_pay_usd"]
    numeric = df[seg_cols].fillna(df[seg_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    k = st.slider("Choose number of clusters", 2, 8, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, clusters)
    st.write(f"Silhouette score: **{sil:.2f}**")

    umap_df = pd.DataFrame(dict(UMAP1=X_umap[:, 0],
                                UMAP2=X_umap[:, 1],
                                cluster=clusters.astype(str)))
    fig = px.scatter(umap_df, x="UMAP1", y="UMAP2",
                     color="cluster", title="UMAP projection of user segments",
                     height=600)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# 3 · FLAVOR & MOOD  (Association Rules)
# ==================================================================
with tabs[2]:
    st.header("🎨 Flavor & Mood Pairing")

    flavor_lists = df["fav_flavor_categories_ranked"].fillna("").apply(
        lambda x: [f.strip() for f in x.split(",") if f.strip()])
    all_flavors = sorted({f for sub in flavor_lists for f in sub})
    flavor_df = pd.DataFrame([
        {flav: (flav in lst) for flav in all_flavors} for lst in flavor_lists
    ])

    freq_items = apriori(flavor_df, min_support=0.05, use_colnames=True)
    rules = (association_rules(freq_items, metric="lift", min_threshold=1.0)
             .sort_values("lift", ascending=False).head(20))

    st.subheader("Top association rules (support ≥ 5 %)")
    st.dataframe(rules[["antecedents", "consequents",
                        "support", "confidence", "lift"]],
                 use_container_width=True)

# ==================================================================
# 4 · CHURN & REFERRAL  (Classification)
# ==================================================================
with tabs[3]:
    st.header("⚠️ Churn & Referral Radar")

    churn_df = df.dropna(subset=["overall_interest_nps"]).copy()
    churn_df["promoter"] = (churn_df["overall_interest_nps"] >= 9).astype(int)
    features = ["interest_gamification_points", "likelihood_refer_friends",
                "flavor_boredom", "freq_seek_recommendations"]
    X = churn_df[features].fillna(0)
    y = churn_df["promoter"]

    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

    st.subheader("Predict promoter probability")
    user_input = {f: st.slider(f, 0, 10, 5) for f in features}
    proba = clf.predict_proba(pd.DataFrame([user_input]))[0, 1]
    st.metric("Promoter probability", f"{proba:.2%}")

# ==================================================================
# 5 · DATA-TRUST CONSOLE  (Clustering)
# ==================================================================
with tabs[4]:
    st.header("🔒 Data-Trust Console")

    trust_cols = ["comfort_sharing_data", "share_mood_data_comfort",
                  "importance_data_control"]
    trust_df = df[trust_cols].fillna(df[trust_cols].median())

    scaler = StandardScaler()
    trust_scaled = scaler.fit_transform(trust_df)

    k_trust = 3
    kmeans_trust = KMeans(n_clusters=k_trust, random_state=42, n_init="auto")
    trust_labels = kmeans_trust.fit_predict(trust_scaled)
    df["trust_tier"] = trust_labels

    st.subheader("Distribution of trust tiers")
    st.bar_chart(df["trust_tier"].value_counts().sort_index())
    st.markdown("**Tier 0** = low trust • **Tier 1** = medium • **Tier 2** = high")

# ==================================================================
# 6 · STATISTICAL INSIGHTS  (Top-10 Charts)
# ==================================================================
with tabs[5]:
    st.header("📊 Statistical Insights")

    charts = [
        px.histogram(df, x="age", nbins=20, title="Age distribution"),
        px.scatter(df, x="monthly_spend_usd", y="willingness_to_pay_usd",
                   trendline="ols", title="Monthly spend vs. willingness to pay"),
        px.scatter(df, x="flavor_boredom", y="openness_new_flavors",
                   title="Flavor boredom vs. openness"),
        px.histogram(df, x="nicotine_strength_mgml",
                     title="Nicotine strength (mg/ml)"),
        px.pie(df, names="device_type", title="Device type split"),
        px.histogram(df, x="weekly_consumption_ml",
                     title="Weekly consumption (ml)"),
        px.histogram(df, x="likelihood_refer_friends",
                     title="Referral likelihood score"),
        px.pie(df, names="self_report_mood_spectrum",
               title="Self-reported mood spectrum"),
        px.box(df, x="preferred_box_size_pods", y="monthly_spend_usd",
               title="Monthly spend by preferred box size"),
        px.histogram(df, x="peak_vape_time", title="Peak vape times"),
    ]
    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)
