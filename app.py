
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title='VaporIQ Analytics', layout='wide')

# ----- Styling -----
st.markdown(
    """
    <style>
        .main {background-color: #f4f2ff;}
        .reportview-container .markdown-text-container {color:#3c3c3c;}
        .sidebar .sidebar-content {background-color:#ebe9f7;}
        footer:after {content:'VaporIQ Dashboard – © 2025';display:block;position:relative;color:#999;padding:5px;text-align:center;}
    </style>
    """, unsafe_allow_html=True)

# ----- Data Loading -----
@st.cache_data(show_spinner=False)
def load_data(path: str):
    return pd.read_csv(path)

DEFAULT_PATH = 'vaporiq_synthetic_v2.csv'
st.sidebar.header('Dataset')
data_file = st.sidebar.file_uploader('Upload CSV (optional)', type=['csv'])
if data_file:
    df = load_data(data_file)
else:
    df = load_data(DEFAULT_PATH)

st.sidebar.success(f'Data rows: {df.shape[0]} | columns: {df.shape[1]}')

# Utility: identify column types
def get_numeric_cols(data):
    return data.select_dtypes(include=np.number).columns.tolist()

def get_categorical_cols(data):
    return [c for c in data.columns if data[c].dtype == 'object' or data[c].nunique() <= 20]

# ----- Tabs -----
tabs = st.tabs(['Data‑Viz', 'Classification', 'Regression', 'Clustering', 'Association Rules'])

# ---------- Data‑Viz ----------
with tabs[0]:
    st.header('Exploratory Data Visualisation')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Column Distributions')
        column = st.selectbox('Pick a column', df.columns)
        if df[column].dtype == 'object':
            counts = df[column].value_counts().head(20)
            fig = px.bar(counts, x=counts.index, y=counts.values, labels={'x':column,'y':'count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df, x=column, nbins=30)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Correlation Heatmap')
        num_cols = get_numeric_cols(df)
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Not enough numeric columns.')

# ---------- Classification ----------
with tabs[1]:
    st.header('Classification Models')
    target_options = [c for c in df.columns if df[c].nunique() <= 10]
    if not target_options:
        st.warning('No categorical/binary target found with ≤10 unique values.')
    else:
        target = st.selectbox('Target', target_options, key='clf_target')
        feature_candidates = [c for c in df.columns if c != target]
        predictors = st.multiselect('Predictor features', feature_candidates, default=feature_candidates[:5], key='clf_feats')
        test_size = st.slider('Test size (%%)', 10, 40, 20, key='clf_split')/100
        grid_toggle = st.checkbox('Enable GridSearchCV (may be slow)', value=False, key='gs')

        X = pd.get_dummies(df[predictors], drop_first=True)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        models = {
            'KNN': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GBRT': GradientBoostingClassifier(random_state=42)
        }
        param_grids = {
            'KNN': {'n_neighbors':[3,5,7]},
            'DecisionTree': {'max_depth':[None,5,10]},
            'RandomForest': {'n_estimators':[100,200], 'max_depth':[None,10]},
            'GBRT': {'n_estimators':[100,200], 'learning_rate':[0.05,0.1]}
        }

        results = []
        roc_data = []

        for name, model in models.items():
            if grid_toggle:
                grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_weighted', n_jobs=-1)
                grid.fit(X_train, y_train)
                best = grid.best_estimator_
            else:
                best = model.fit(X_train, y_train)

            y_pred = best.predict(X_test)
            row = {
                'Model':name,
                'Accuracy':accuracy_score(y_test, y_pred),
                'Precision':precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall':recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1':f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            results.append(row)

            # ROC only for binary targets
            if y.nunique() == 2:
                if hasattr(best, 'predict_proba'):
                    proba = best.predict_proba(X_test)[:,1]
                else:
                    proba = best.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr,tpr)
                roc_data.append((name, fpr, tpr, roc_auc))

        st.subheader('Metrics')
        st.dataframe(pd.DataFrame(results).set_index('Model').style.format('{:.3f}'))

        if y.nunique() == 2 and roc_data:
            st.subheader('ROC Curves')
            fig = go.Figure()
            for name, fpr, tpr, roc_auc in roc_data:
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)

# ---------- Regression ----------
with tabs[2]:
    st.header('Regression Models')
    numeric_cols = get_numeric_cols(df)
    if len(numeric_cols) < 2:
        st.info('Need at least 2 numeric columns.')
    else:
        target_reg = st.selectbox('Numeric target', numeric_cols, key='reg_target')
        reg_features = st.multiselect('Predictors', [c for c in numeric_cols if c != target_reg], default=[c for c in numeric_cols if c != target_reg][:5], key='reg_feats')
        test_size_reg = st.slider('Test size (%%)', 10, 40, 20, key='reg_split')/100

        Xr = df[reg_features]
        yr = df[target_reg]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=test_size_reg, random_state=42)

        regs = {
            'Linear': LinearRegression(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'GBRT': GradientBoostingRegressor(random_state=42)
        }
        reg_results = []
        for name, reg in regs.items():
            reg.fit(Xr_train, yr_train)
            preds = reg.predict(Xr_test)
            reg_results.append({
                'Model': name,
                'R2': r2_score(yr_test, preds),
                'MAE': mean_absolute_error(yr_test, preds),
                'RMSE': mean_squared_error(yr_test, preds, squared=False)
            })

        st.subheader('Metrics')
        st.dataframe(pd.DataFrame(reg_results).set_index('Model').style.format('{:.3f}'))

        selected_model = st.selectbox('Plot predictions for', list(regs.keys()), key='pred_plot')
        model_obj = regs[selected_model]
        preds_full = model_obj.predict(Xr)
        fig2 = px.scatter(x=yr, y=preds_full, labels={'x':'Actual','y':'Predicted'}, trendline='ols')
        fig2.update_traces(marker={'opacity':0.6})
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Clustering ----------
with tabs[3]:
    st.header('K‑Means Clustering')
    cat_cols = get_categorical_cols(df)
    num_cols = get_numeric_cols(df)
    selected_cols = st.multiselect('Pick features (numeric & one‑hot encoded)', df.columns, default=num_cols[:3]+cat_cols[:1], key='clus_feats')
    if selected_cols:
        data_clu = pd.get_dummies(df[selected_cols], drop_first=True)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clu)

        k_range = st.slider('Max k for elbow', 2, 15, 10, key='k_max')
        distortions = []
        Ks = range(1, k_range+1)
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(data_scaled)
            distortions.append(km.inertia_)

        fig3 = px.line(x=list(Ks), y=distortions, markers=True, labels={'x':'k','y':'Inertia'})
        st.plotly_chart(fig3, use_container_width=True)

        k_choose = st.number_input('Choose k to cluster', min_value=2, max_value=k_range, value=3, step=1, key='k_choose')
        km_final = KMeans(n_clusters=int(k_choose), random_state=42, n_init='auto').fit(data_scaled)
        st.subheader('Cluster Counts')
        st.write(pd.Series(km_final.labels_).value_counts().rename('count'))

# ---------- Association Rules ----------
with tabs[4]:
    st.header('Association Rule Mining')
    list_like_cols = ['fav_flavor_categories_ranked','top_subscription_value_ranked','cancel_triggers_open']
    present_cols = [c for c in list_like_cols if c in df.columns]
    if not present_cols:
        st.warning('No list‑like columns found.')
    else:
        st.write('Using columns:', ', '.join(present_cols))
        # Build transaction list
        transactions = []
        for _, row in df[present_cols].iterrows():
            items = []
            for c in present_cols:
                val = row[c]
                if pd.isna(val):
                    continue
                if isinstance(val, str):
                    items.extend([i.strip() for i in val.split(',') if i.strip()])
            transactions.append(items)

        # One‑hot encode
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        trans_df = pd.DataFrame(te_ary, columns=te.columns_)

        min_support = st.slider('Min support', 0.01, 0.5, 0.05, 0.01)
        freq = apriori(trans_df, min_support=min_support, use_colnames=True)
        if freq.empty:
            st.info('No frequent itemsets at this support.')
        else:
            min_conf = st.slider('Min confidence', 0.1, 1.0, 0.3, 0.05)
            min_lift = st.slider('Min lift', 1.0, 10.0, 1.2, 0.1)
            rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
            rules = rules[rules['lift'] >= min_lift]
            st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].sort_values('lift', ascending=False).head(50))
