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
import networkx as nx
import itertools
import pathlib

# Look for vaporiq_data.csv **next to** app.py
DATA_PATH = pathlib.Path(__file__).with_name("vaporiq_data.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    """Read the survey CSV, or prompt upload if it’s absent."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    # Friendly fallback—let the user upload the file
    st.warning(
        f"⚠️ Couldn’t find {DATA_PATH.name}. "
        "Upload the CSV manually and I’ll use that copy."
    )
    uploaded = st.file_uploader("Upload vaporiq_data.csv", type="csv")
    if uploaded is not None:
        return pd.read_csv(uploaded)

    # Stop the app until a file is provided
    st.stop()

df = load_data()

st.set_page_config(page_title="VaporIQ Dashboard", layout="wide")
st.title("🚀 VaporIQ – Hyper‑Personalized Vape Subscription Intelligence")

tabs = st.tabs([
    "Personal Pricing",
    "Segment Explorer",
    "Flavor & Mood",
    "Churn & Referral",
    "Data‑Trust Console",
    "Statistical Insights"
])

# 1. PERSONAL PRICING
with tabs[0]:
    st.header("💸 Personal Pricing")
    target = "willingness_to_pay_usd"
    num_cols = ["age", "weekly_consumption_ml", "nicotine_strength_mgml", "monthly_spend_usd"]
    cat_cols = ["country", "device_type", "primary_vape_motivation"]
    df_mod = df.dropna(subset=[target])

    X = df_mod[num_cols + cat_cols]
    y = df_mod[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    model = Pipeline([
        ("pre", pre),
        ("reg", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ])
    model.fit(X, y)

    st.subheader("Predict user price ceiling")
    sample = {}
    for c in num_cols:
        sample[c] = st.number_input(c, float(df[c].min()), float(df[c].max()), float(df[c].mean()))
    for c in cat_cols:
        sample[c] = st.selectbox(c, sorted(df[c].dropna().unique()))

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([sample]))[0]
        st.metric("Estimated willingness to pay (USD)", f"{pred:,.2f}")

# 2. SEGMENT EXPLORER
with tabs[1]:
    st.header("🔍 Segment Explorer")
    seg_cols = ["age", "weekly_consumption_ml", "monthly_spend_usd", "willingness_to_pay_usd"]
    numeric = df[seg_cols].fillna(df[seg_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    k = st.slider("Choose number of clusters", 2, 8, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, clusters)
    st.write(f"Silhouette score: {sil:.2f}")

    umap_df = pd.DataFrame(dict(UMAP1=X_umap[:,0], UMAP2=X_umap[:,1], cluster=clusters.astype(str)))
    fig = px.scatter(umap_df, x="UMAP1", y="UMAP2", color="cluster", title="UMAP projection of user segments")
    st.plotly_chart(fig, use_container_width=True)

# 3. FLAVOR & MOOD
with tabs[2]:
    st.header("🎨 Flavor & Mood Pairing")
    from mlxtend.frequent_patterns import apriori, association_rules

    # Transform ranked flavor list into one‑hot columns
    flavor_lists = df["fav_flavor_categories_ranked"].fillna("").apply(lambda x: [f.strip() for f in x.split(",") if f.strip()])
    all_flavors = sorted({f for sub in flavor_lists for f in sub})
    flavor_df = pd.DataFrame([{flav: (flav in lst) for flav in all_flavors} for lst in flavor_lists])

    freq_items = apriori(flavor_df, min_support=0.05, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.0).sort_values("lift", ascending=False).head(20)

    st.subheader("Top association rules (support ≥ 5 %)")
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# 4. CHURN & REFERRAL
with tabs[3]:
    st.header("⚠️ Churn & Referral Radar")
    churn_target = "overall_interest_nps"
    churn_df = df.dropna(subset=[churn_target])
    churn_df["promoter"] = (churn_df[churn_target] >= 9).astype(int)

    features = ["interest_gamification_points", "likelihood_refer_friends",
                "flavor_boredom", "freq_seek_recommendations"]
    X = churn_df[features].fillna(0)
    y = churn_df["promoter"]

    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

    st.subheader("Predict promoter probability")
    user_input = {f: st.slider(f, 0, 10, 5) for f in features}
    proba = clf.predict_proba(pd.DataFrame([user_input]))[0,1]
    st.metric("Promoter probability", f"{proba:.2%}")

# 5. DATA‑TRUST CONSOLE
with tabs[4]:
    st.header("🔒 Data‑Trust Console")
    trust_cols = ["comfort_sharing_data", "share_mood_data_comfort", "importance_data_control"]
    trust_df = df[trust_cols].fillna(df[trust_cols].median())
    scaler = StandardScaler()
    trust_scaled = scaler.fit_transform(trust_df)

    k_trust = 3
    kmeans_trust = KMeans(n_clusters=k_trust, random_state=42, n_init="auto").fit(trust_scaled)
    trust_labels = kmeans_trust.labels_
    df["trust_tier"] = trust_labels

    count_tiers = df["trust_tier"].value_counts().sort_index()
    st.bar_chart(count_tiers)
    st.write("Tier 0 = low trust, Tier 1 = medium, Tier 2 = high")

# 6. STATISTICAL INSIGHTS
with tabs[5]:
    st.header("📊 Statistical Insights")
    st.write("Top‑10 quick graphs")

    graphs = []
    # 1. Age distribution
    graphs.append(px.histogram(df, x="age", nbins=20, title="Age distribution"))
    # 2. Monthly spend vs. willingness
    graphs.append(px.scatter(df, x="monthly_spend_usd", y="willingness_to_pay_usd", trendline="ols", title="Spend vs. Willingness to Pay"))
    # 3. Flavor boredom vs. openness
    graphs.append(px.scatter(df, x="flavor_boredom", y="openness_new_flavors", title="Boredom vs. Openness"))
    # 4. Nicotine strength histogram
    graphs.append(px.histogram(df, x="nicotine_strength_mgml", title="Nicotine strength"))
    # 5. Device type pie
    graphs.append(px.pie(df, names="device_type", title="Device types"))
    # 6. Weekly consumption histogram
    graphs.append(px.histogram(df, x="weekly_consumption_ml", title="Weekly consumption (ml)"))
    # 7. Referral likelihood distribution
    graphs.append(px.histogram(df, x="likelihood_refer_friends", title="Referral likelihood"))
    # 8. Mood spectrum pie
    graphs.append(px.pie(df, names="self_report_mood_spectrum", title="Mood spectrum"))
    # 9. Box size vs. spend
    graphs.append(px.box(df, x="preferred_box_size_pods", y="monthly_spend_usd", title="Spend by Box Size"))
    # 10. Peak time bar
    graphs.append(px.histogram(df, x="peak_vape_time", title="Peak vape times"))

    for fig in graphs:
        st.plotly_chart(fig, use_container_width=True)
