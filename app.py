# ───────── VaporIQ Dashboard • v8 (New Dataset) ─────────
# Compatible with: vaporiq_synthetic.csv (44 cols) & flavor_trends_user_based.csv
# Tabs: Data Visualization, TasteDNA, Forecasting, Micro-Batch
# Added:
#  • Richer core feature set leveraging additional survey columns
#  • Model target updated to SubscribeIntent (current_vape_subscription)
#  • Feature‑importance visual for tree‑based models
#  • Visuals updated to use new numeric preference columns
# --------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
from pathlib import Path
import base64, textwrap, warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score,
                             roc_curve, auc, r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

# ─────────────────────  Page & Gradient  ─────────────────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ─────────  Galaxy star‑field (body::before)  ─────────
star_path = Path(__file__).with_name("starfield.png")
if star_path.exists():
    star_b64 = base64.b64encode(star_path.read_bytes()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
        body::before {{
            content:"";
            position:fixed; inset:0; z-index:-4;
            pointer-events:none;
            background:url("data:image/png;base64,{star_b64}") repeat;
            background-size:600px;
            opacity:.35;
            animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{
          0%   {{transform:translate3d(0,0,0);}}
          100% {{transform:translate3d(-2000px,1500px,0);}}
        }}
        .smoke-layer    {{animation:smokeFlow 210s linear infinite;  opacity:.25;}}
        .smoke-layer-2  {{animation:smokeFlowR 280s linear infinite; opacity:.15;}}
        @keyframes smokeFlow  {{0%{{background-position:0 0}} 100%{{background-position:1600px 0}}}}
        @keyframes smokeFlowR {{0%{{background-position:0 0}} 100%{{background-position:-1600px 0}}}}
        </style>
    """), unsafe_allow_html=True)
else:
    st.sidebar.error("⚠️ `starfield.png` not found – galaxy backdrop disabled.")

# Inject smoke divs
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)

# ─────────  Watermark bottle  ─────────
wm_path = Path(__file__).with_name("vape_watermark.png")
if wm_path.exists():
    with open(wm_path,"rb") as f:
        wm_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"<img src='data:image/png;base64,{wm_b64}' style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>",
                unsafe_allow_html=True)

# ─────────────────────  Data  ─────────────────────
@st.cache_data
def load_data():
    users = pd.read_csv("vaporiq_synthetic.csv")  # renamed upload
    trends = pd.read_csv("flavor_trends_user_based.csv")

    # Harmonise column names with legacy code
    rename_map = {
        "age": "Age",
        "gender": "Gender",
        "weekly_consumption_ml": "PodsPerWeek",
        "fav_flavor_categories_ranked": "FlavourFamilies",
        "current_vape_subscription": "SubscribeIntent"
    }
    users = users.rename(columns=rename_map)

    # Ensure binary target (1=subscribed/likely, 0 otherwise)
    if users["SubscribeIntent"].dtype == object:
        users["SubscribeIntent"] = users["SubscribeIntent"].map({"Yes": 1, "No": 0})

    return users, trends

users_df, trends_df = load_data()

# Expanded core feature set for modelling
core = [
    "Age", "years_vaping", "PodsPerWeek", "nicotine_strength_mgml",
    "openness_new_flavors", "value_exclusivity", "flavor_boredom",
    "monthly_spend_usd", "preferred_box_size_pods", "interest_gamification_points",
    "overall_interest_nps"
]

num_core = [c for c in core if c in users_df.columns]
users_df[num_core] = users_df[num_core].apply(pd.to_numeric, errors="coerce")

# ─────────────────────  Tabs  ─────────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# =================== 1. Data‑Viz TAB ===================
with viz:
    st.header("📊 Data Visualization Explorer")

    genders = st.sidebar.multiselect("Gender filter",
                 users_df["Gender"].unique().tolist(),
                 default=users_df["Gender"].unique().tolist())
    df = users_df[users_df["Gender"].isin(genders)]

    if df.empty:
        st.warning("No rows match current filters — tweak sidebar options.")
        st.stop()

    # 1 Density heatmap Age vs Pods
    hex_fig = px.density_heatmap(
        df, x="Age", y="PodsPerWeek",
        nbinsx=30, nbinsy=15,
        color_continuous_scale="magma",
        title="Density of Consumption by Age")
    st.plotly_chart(hex_fig, use_container_width=True)

    # 2 Scatter Pods vs Age
    fig, ax = plt.subplots(); sns.scatterplot(data=df, x="Age", y="PodsPerWeek", ax=ax, alpha=.6)
    st.pyplot(fig); plt.close(fig)

    # 3 Correlation heat‑map (selected numeric columns)
    corr_cols = ["Age", "PodsPerWeek", "nicotine_strength_mgml", "monthly_spend_usd"]
    corr_df = df[corr_cols].corr()
    fig, ax = plt.subplots(); sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig); plt.close(fig)

    # 4 Bar: flavour family counts
    flat = df["FlavourFamilies"].str.get_dummies(sep=", ").sum().sort_values(ascending=False)
    st.bar_chart(flat)

    # 5 Top‑3 flavour trends (user‑based)
    top3 = trends_df.drop(columns="Date").mean().nlargest(3).index
    st.plotly_chart(px.line(trends_df, x="Date", y=top3, title="Top‑3 Flavour Trends"),
                    use_container_width=True)

    # 6 Rugplots: openness & boredom
    fig, ax = plt.subplots()
    sns.rugplot(df["openness_new_flavors"], height=.1, color="g", ax=ax, label="Openness")
    sns.rugplot(df["flavor_boredom"], height=.1, color="r", ax=ax, label="Boredom")
    ax.legend(); st.pyplot(fig); plt.close(fig)

  # --- NEW box-and-whisker: Monthly Spend by Gender -----------------
    if "monthly_spend_usd" in df.columns:
        fig,ax=plt.subplots();sns.boxplot(data=df,x="Gender",y="monthly_spend_usd",ax=ax)
        ax.set_xlabel("Gender");ax.set_ylabel("Monthly Spend (USD)")
        st.pyplot(fig);plt.close(fig)
        st.caption("Box‑and‑whisker shows spending dispersion by gender – pinpoints high‑value segments and outliers for VIP marketing.")
  
   with st.expander("Key Insights & Rationale"):
        dom_gender = df["Gender"].value_counts(normalize=True).idxmax()
        fast_flav = trends_df.drop(columns="Date").mean().idxmax()
        spend_gap = df.groupby("Gender")["monthly_spend_usd"].median().to_dict() if "monthly_spend_usd" in df.columns else {}
        st.markdown(f"• **Dominant gender:** {dom_gender}")
        st.markdown(f"• **Top buzzing flavour family:** {fast_flav}")
        if spend_gap:
            st.markdown(f"• **Median spend gap:** {spend_gap}")
        st.markdown("The new box‑plot highlights skewness and outliers, guiding premium bundle design and loyalty tiers.")

# =================== 2. TasteDNA TAB ===================
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    m_mode = st.radio("Mode", ["Classification", "Clustering"], horizontal=True)

    if m_mode == "Classification":
        algo = st.selectbox("Classifier", ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"])
        tune = st.checkbox("GridSearch (5‑fold F1)", False)

        X, y = users_df[num_core].fillna(0), users_df["SubscribeIntent"].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)

        base = {"KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)}[algo]

        grid = {"KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                "Decision Tree": {"max_depth": [None, 5, 10]},
                "Random Forest": {"n_estimators": [200, 300, 400], "max_depth": [None, 10]},
                "Gradient Boosting": {"n_estimators": [200, 300], "learning_rate": [0.05, 0.1]}}[algo]

        if tune:
            gs = GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1).fit(X_tr, y_tr)
            model, best = gs.best_estimator_, gs.best_params_
        else:
            model, best = base.fit(X_tr, y_tr), None

        y_pred = model.predict(X_te)
        prec, rec = precision_score(y_te, y_pred), recall_score(y_te, y_pred)
        acc, f1 = accuracy_score(y_te, y_pred), f1_score(y_te, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{prec:.2f}")
        m2.metric("Recall",   f"{rec:.2f}")
        m3.metric("Accuracy", f"{acc:.2f}")
        m4.metric("F1",       f"{f1:.2f}")

        fig, ax = plt.subplots(); sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig); plt.close(fig)

        # Feature importance for tree‑based models
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=num_core).sort_values(ascending=False)[:10]
            fig_imp, ax_imp = plt.subplots(); sns.barplot(x=imp.values, y=imp.index, ax=ax_imp)
            ax_imp.set_title("Top Feature Importances"); st.pyplot(fig_imp); plt.close(fig_imp)

        with st.expander("Key Insights"):
            if best: st.markdown(f"- GridSearch best params: `{best}`")
            st.markdown(f"- Precision {prec:.2f}, Recall {rec:.2f}, F1 {f1:.2f}")

    else:  # Clustering
        k = st.slider("k clusters", 2, 10, 4)
        X_scaled = MinMaxScaler().fit_transform(users_df[num_core].fillna(0))
     # NEW: Elbow method chart
        ks = range(2,11)
        inertias = [KMeans(i,random_state=42,n_init='auto').fit(X_scaled).inertia_ for i in ks]
        fig_elbow,ax_elbow=plt.subplots();ax_elbow.plot(list(ks),inertias,'o-');ax_elbow.set_xlabel("k");ax_elbow.set_ylabel("Inertia");ax_elbow.set_title("Elbow Method for k choice");st.pyplot(fig_elbow);plt.close(fig_elbow)        
        km = KMeans(k, random_state=42, n_init="auto").fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        users_df["Cluster"] = km.labels_
        st.metric("Silhouette", f"{sil:.3f}")
        st.dataframe(users_df.groupby("Cluster")[num_core].mean().round(2))

# =================== 3. Forecast TAB ===================
with forecast_tab:
    st.header("📈 Forecasting")
    flavour = st.selectbox("Flavour signal", trends_df.columns[1:])
    reg_name = st.selectbox("Regressor", ["Linear", "Ridge", "Lasso", "Decision Tree"])
    reg_map = {"Linear": LinearRegression(),
               "Ridge": Ridge(alpha=1.0),
               "Lasso": Lasso(alpha=0.01),
               "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)}
    reg = reg_map[reg_name]

    X = np.arange(len(trends_df)).reshape(-1, 1); y = trends_df[flavour].values
    split = int(.8 * len(X)); reg.fit(X[:split], y[:split]); y_pred = reg.predict(X[split:])
    r2 = r2_score(y[split:], y_pred); rmse = np.sqrt(mean_squared_error(y[split:], y_pred))

    st.metric("R²", f"{r2:.3f}"); st.metric("RMSE", f"{rmse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[split:], y_pred, alpha=.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax.set_xlabel("Actual Mentions"); ax.set_ylabel("Predicted Mentions")
    st.pyplot(fig); plt.close(fig)

    with st.expander("Key Insights"):
        slopes = {c: np.polyfit(np.arange(len(trends_df)), trends_df[c], 1)[0] for c in trends_df.columns[1:]}
        st.markdown(f"- Regressor **{reg_name}** → R² {r2:.2f}, RMSE {rmse:.2f}")
        st.markdown(f"- Steepest flavour slope: **{max(slopes, key=slopes.get)}**")

# =================== 4. Apriori TAB ===================
with rules_tab:
    st.header("🧩 Apriori Explorer")
    sup = st.slider("Support", 0.01, 0.4, 0.05, 0.01); conf = st.slider("Confidence", 0.05, 1.0, 0.3, 0.05)

    basket = users_df["FlavourFamilies"].str.get_dummies(sep=", ").astype(bool)
    basket = pd.concat([
        basket,
        pd.get_dummies(users_df["device_type"], prefix="Dev", dtype=bool),
        pd.get_dummies(users_df["Gender"], prefix="Gen", dtype=bool)
    ], axis=1)

    freq = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)

    if rules.empty:
        st.warning("No rules under thresholds.")
        best = None
    else:
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules)
        best = rules.iloc[0]

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- Best rule: {best['antecedents']} → {best['consequents']} (lift {best['lift']:.2f})")
        st.markdown(f"- Support ≥ {sup:.2f}, Confidence ≥ {conf:.2f}")
