import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             silhouette_score, r2_score, mean_squared_error,
                             mean_absolute_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_data(src: str):
    """Load CSV from local path or GitHub RAW URL."""
    return pd.read_csv(src)

def train_classifiers(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }
    metrics, rocs, cms = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
        metrics.append(dict(Model=name,
                            Accuracy=accuracy_score(y_test,pred),
                            Precision=precision_score(y_test,pred,zero_division=0),
                            Recall=recall_score(y_test,pred,zero_division=0),
                            F1=f1_score(y_test,pred,zero_division=0)))
        cms[name]=confusion_matrix(y_test,pred)
        if prob is not None:
            fpr,tpr,_=roc_curve(y_test,prob)
            rocs[name]=(fpr,tpr,auc(fpr,tpr))
    return pd.DataFrame(metrics), rocs, cms, scaler, models

def kmeans_cluster(df, features, k):
    X = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    df_out = df.loc[X.index].copy()
    df_out['cluster'] = labels
    centers = pd.DataFrame(km.cluster_centers_, columns=features)
    return df_out, inertia, sil, centers

def association_rules_top(df, cols, min_sup, min_conf):
    basket = pd.DataFrame()
    for c in cols:
        basket = pd.concat([basket, df[c].str.get_dummies(sep=',')], axis=1)
    basket = basket.groupby(level=0, axis=1).max()
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
    return rules.sort_values('confidence', ascending=False).head(10)

def train_regressors(df, target, scaler_choice='robust'):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler() if scaler_choice=='robust' else MinMaxScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    grids = {"Ridge":{"alpha":[0.1,1,10]},
             "Lasso":{"alpha":[0.001,0.01,0.1]},
             "Decision Tree":{"max_depth":[None,5,10,20]}}
    results, tuned = [], {}
    for name, model in models.items():
        if name in grids:
            gs = GridSearchCV(model, grids[name], cv=5, scoring='r2').fit(X_train,y_train)
            best = gs.best_estimator_
        else:
            best = model.fit(X_train,y_train)
        tuned[name]=best
        pred = best.predict(X_test)
        results.append(dict(Model=name,
                            R2=r2_score(y_test,pred),
                            RMSE=mean_squared_error(y_test,pred,squared=False),
                            MAE=mean_absolute_error(y_test,pred)))
    return pd.DataFrame(results), tuned, scaler

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="VaporIQ Dashboard", layout="wide")
st.title("📊 VaporIQ Analytics (All‑in‑One)")

# Data source
st.sidebar.header("Data")
csv_src = st.sidebar.text_input("CSV path / RAW URL", "vaporiq_synthetic.csv")
df = load_data(csv_src)
st.sidebar.success(f"Loaded {df.shape[0]:,} rows.")

# Tabs
tabs = st.tabs(["Classification","Clustering","Association Rules","Regression","Data Viz"])

# ----- Classification -----
with tabs[0]:
    st.header("🧩 Classification")
    targets = [c for c in df.columns if df[c].nunique()<=10 and df[c].dtype!='object']
    target = st.selectbox("Target label", targets)
    if st.button("Train models"):
        met, rocs, cms, scaler, fitted = train_classifiers(df, target)
        st.dataframe(met.style.format({"Accuracy":"{:.2%}","Precision":"{:.2%}",
                                       "Recall":"{:.2%}","F1":"{:.2%}"}))
        fig, ax = plt.subplots()
        for n,(fpr,tpr,aucv) in rocs.items():
            ax.plot(fpr,tpr,label=f"{n} (AUC={aucv:.2f})")
        ax.plot([0,1],[0,1],'k--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        st.pyplot(fig)
        algo = st.selectbox("Confusion matrix for:", list(cms.keys()))
        if algo:
            st.dataframe(pd.DataFrame(cms[algo], index=["True 0","True 1"],
                                       columns=["Pred 0","Pred 1"]))
    st.subheader("Batch prediction")
    up = st.file_uploader("Upload unlabeled CSV")
    if up:
        nd = pd.read_csv(up)
        alg = st.selectbox("Predict with model:", list(fitted.keys()))
        if st.button("Predict"):
            nd["prediction"] = fitted[alg].predict(scaler.transform(nd))
            buf = BytesIO(); nd.to_csv(buf,index=False)
            st.download_button("Download predictions", buf.getvalue(), "predictions.csv")

# ----- Clustering -----
with tabs[1]:
    st.header("👥 Clustering")
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = st.multiselect("Features", numcols, default=numcols[:3])
    k = st.slider("k clusters", 2, 10, 4)
    if st.button("Run K‑Means"):
        dc, inertia, sil, centers = kmeans_cluster(df, feats, k)
        st.write(f"SSE: {inertia:.0f} | Silhouette: {sil:.2f}")
        st.dataframe(centers)
        st.dataframe(dc.groupby('cluster')[feats].mean())
        buf=BytesIO(); dc.to_csv(buf,index=False)
        st.download_button("Download labeled data", buf.getvalue(), "clusters.csv")

# ----- Association -----
with tabs[2]:
    st.header("🔗 Association Rules")
    basket_cols = [c for c in df.columns if df[c].astype(str).str.contains(',').any()]
    chos = st.multiselect("Basket columns", basket_cols, default=basket_cols[:2])
    ms = st.slider("Min support",0.01,0.2,0.05,0.01)
    mc = st.slider("Min confidence",0.1,0.9,0.3,0.05)
    if st.button("Run Apriori"):
        rules = association_rules_top(df, chos, ms, mc)
        st.dataframe(rules)

# ----- Regression -----
with tabs[3]:
    st.header("📈 Regression")
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    tgt = st.selectbox("Numeric target", numcols, index=numcols.index("monthly_spend_usd") if "monthly_spend_usd" in numcols else 0)
    scaler_opt = st.radio("Scaler", ["robust","minmax"], horizontal=True)
    if st.button("Train regressors"):
        res, models, scalerR = train_regressors(df[numcols].dropna(), tgt, scaler_opt)
        st.dataframe(res.style.format({"R2":"{:.2f}","RMSE":"{:.1f}","MAE":"{:.1f}"}))
        best = res.iloc[0]["Model"]
        X = df[numcols].drop(columns=[tgt])
        y = df[tgt]
        y_pred = models[best].predict(scalerR.transform(X))
        fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual","y":"Predicted"}, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

# ----- Data Viz -----
with tabs[4]:
    st.header("🖼️ Data Visualisation Insights")
    fig1 = px.box(df, x="flavor_boredom", y="monthly_spend_usd")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.histogram(df, x="nicotine_strength_mgml", color="intend_reduce_nicotine", barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("Add up to 10 complex insights here…")