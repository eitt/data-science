import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time  # To time model fits

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.datasets import make_classification

# NEW: Imports for Tab 5 (using statsmodels)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import statsmodels.api as sm
except ImportError:
    st.error("Please ensure 'statsmodels' is installed.")

# --- Page Configuration ---
st.set_page_config(
    page_title="Understanding Regression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.header("Global Controls")
    
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if "n_samples" not in st.session_state:
        st.session_state.n_samples = 100
    if "noise" not in st.session_state:
        st.session_state.noise = 0.5

    def update_controls():
        st.session_state.seed = st.session_state._seed
        st.session_state.n_samples = st.session_state._n_samples
        st.session_state.noise = st.session_state._noise

    st.number_input("Random Seed", value=st.session_state.seed, key="_seed", on_change=update_controls)
    st.slider("Sample Size", 20, 300, value=st.session_state.n_samples, key="_n_samples", on_change=update_controls)
    st.slider("Noise (σ)", 0.0, 2.0, value=st.session_state.noise, step=0.05, key="_noise", on_change=update_controls)

    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

    st.header("Key Formulas")
    st.latex(r"\hat{y}_i = \beta_0 + \beta_1 x_i")
    st.latex(r"R^2 = 1 - \frac{\text{SSE}}{\text{SST}}")
    st.latex(r"\text{SSE} = \sum (y_i - \hat{y}_i)^2")
    st.latex(r"\text{MAE} = \frac{1}{n}\sum |e_i|")
    st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n}\sum e_i^2}")
    st.latex(r"p(y=1|x)=\sigma(\beta_0+\beta^\top x)")
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")

# Set random seed for reproducibility
np.random.seed(st.session_state.seed)

# --- Data Caching ---
@st.cache_data
def get_linear_data(n_samples, noise, seed):
    np.random.seed(seed)
    true_b0, true_b1 = 2, 3
    x = np.random.rand(n_samples)
    y = true_b0 + true_b1 * x + np.random.randn(n_samples) * noise
    return x, y, true_b0, true_b1

@st.cache_data
def get_poly_data(n_samples, noise, seed):
    np.random.seed(seed)
    x = np.sort(np.random.rand(n_samples))
    y_true = np.sin(2 * np.pi * x)
    y = y_true + np.random.randn(n_samples) * noise
    return x, y, y_true

@st.cache_data
def get_logistic_data(n_samples, noise, seed):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, flip_y=0.01, class_sep=0.5 + (1.5 - noise),
        random_state=seed,
    )
    return X, y

# Time Series Data Function (Tab 4)
@st.cache_data
def get_timeseries_data(n_samples, noise, seed):
    np.random.seed(seed)
    time_idx = np.arange(n_samples)
    seasonality = np.sin(2 * np.pi * time_idx / 50) * 10
    trend = time_idx * 0.5
    y = trend + seasonality + np.random.randn(n_samples) * (noise * 5)
    df = pd.DataFrame({'time': time_idx, 'y': y})
    df['date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df['time'], unit='D')
    df = df.set_index('date')
    return df

# Time Series Feature Engineering Function (Tab 4)
def create_lagged_features(df, lags):
    df_new = df.copy()
    for i in range(1, lags + 1):
        df_new[f'y_lag_{i}'] = df_new['y'].shift(i)
    df_new = df_new.dropna()
    return df_new

# --- Main App ---
st.title("Understanding Regression: Linear, Polynomial, and Logistic")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Manual Linear Fit", "Train vs Test (Overfitting)", "Logistic Regression", "Time Series (Data Leaks)", "TS Model Playground"]
)

# ==============================================================================
# --- Tab 1: Manual Linear Fit ---
# ==============================================================================
with tab1:
    st.header("Tab 1: Manual Linear Fit")
    st.markdown("Try to minimize SSE manually. Then click 'Compute OLS' to compare.")
    
    x, y, true_b0, true_b1 = get_linear_data(st.session_state.n_samples, st.session_state.noise, st.session_state.seed)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        b0 = st.slider("Intercept (β₀)", -5.0, 5.0, 0.0, 0.1)
        b1 = st.slider("Slope (β₁)", -5.0, 5.0, 1.0, 0.1)
        show_true_line = st.checkbox("Show true underlying line", value=False)
        show_residuals = st.checkbox("Show vertical projections (residuals)", value=True)
        
        y_pred = b0 + b1 * x
        residuals = y - y_pred
        sse = np.sum(residuals**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - (sse / sst) if sst != 0 else 0
        rmse = np.sqrt(sse / len(y))
        
        st.subheader("Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("SSE", f"{sse:.2f}")
        m_col2.metric("RMSE", f"{rmse:.2f}")
        m_col3.metric("R²", f"{r2:.3f}")

        if st.button("Compute OLS Automatically"):
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            st.session_state.ols_b0 = model.intercept_
            st.session_state.ols_b1 = model.coef_[0]
            st.success(f"Optimal OLS Fit: β₀ = {st.session_state.ols_b0:.2f}, β₁ = {st.session_state.ols_b1:.2f}")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data (x, y)'))
        fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Manual Fit (ŷ)', line=dict(color='orange', width=2)))
        
        if show_true_line:
            y_true_line = true_b0 + true_b1 * x
            fig.add_trace(go.Scatter(x=x, y=y_true_line, mode='lines', name='True Line', line=dict(color='blue', dash='dash', width=2)))
        
        if 'ols_b0' in st.session_state:
            y_ols_line = st.session_state.ols_b0 + st.session_state.ols_b1 * x
            fig.add_trace(go.Scatter(x=x, y=y_ols_line, mode='lines', name='OLS Fit', line=dict(color='green', dash='dot', width=3)))

        if show_residuals:
            for i in range(len(x)):
                fig.add_shape(type="line", x0=x[i], y0=y_pred[i], x1=x[i], y1=y[i], line=dict(color="red", width=1, dash="dot"))
        
        fig.update_layout(title="Manual Linear Regression Fit", xaxis_title="x", yaxis_title="y", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data and Residuals")
    df_linear = pd.DataFrame({'x': x, 'y': y, 'ŷ (prediction)': y_pred, 'Residual (eᵢ)': residuals})
    st.dataframe(df_linear.head(), use_container_width=True)

# ==============================================================================
# --- Tab 2: Training vs Testing: Overfitting ---
# ==============================================================================
with tab2:
    st.header("Tab 2: Training vs Testing: Overfitting")
    st.markdown("Training error almost always decreases with complexity, but test error increases after the optimal point.")

    x, y, y_true = get_poly_data(st.session_state.n_samples, st.session_state.noise, st.session_state.seed)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        st.slider("Polynomial Degree", 0, 15, 1, key="poly_degree")
        train_pct = st.slider("Training Set Size", 0.6, 0.9, 0.7, 0.05)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_pct, random_state=st.session_state.seed)
        
        sort_idx = np.argsort(x_train)
        x_train_s, y_train_s = x_train[sort_idx], y_train[sort_idx]
        sort_idx_test = np.argsort(x_test)
        x_test_s, y_test_s = x_test[sort_idx_test], y_test[sort_idx_test]

        metric_choice = st.selectbox("Error Metric", ("RMSE", "MSE", "MAE"))
        standardize = st.checkbox("Standardize x (recommended for high degrees)")

    @st.cache_data
    def get_all_errors(train_pct, _x, _y, _seed, _standardize, _metric):
        x_train_loop, x_test_loop, y_train_loop, y_test_loop = train_test_split(_x, _y, train_size=train_pct, random_state=_seed)
        degrees = list(range(0, 16))
        train_errors, test_errors = [], []

        for deg in degrees:
            steps = [('poly', PolynomialFeatures(degree=deg))]
            if _standardize:
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', LinearRegression()))
            pipeline = Pipeline(steps)
            pipeline.fit(x_train_loop.reshape(-1, 1), y_train_loop)
            y_train_pred = pipeline.predict(x_train_loop.reshape(-1, 1))
            y_test_pred = pipeline.predict(x_test_loop.reshape(-1, 1))
            
            if _metric == "RMSE":
                train_errors.append(np.sqrt(mean_squared_error(y_train_loop, y_train_pred)))
                test_errors.append(np.sqrt(mean_squared_error(y_test_loop, y_test_pred)))
            elif _metric == "MSE":
                train_errors.append(mean_squared_error(y_train_loop, y_train_pred))
                test_errors.append(mean_squared_error(y_test_loop, y_test_pred))
            else:
                train_errors.append(mean_absolute_error(y_train_loop, y_train_pred))
                test_errors.append(mean_absolute_error(y_test_loop, y_test_pred))
        return degrees, train_errors, test_errors

    degrees, train_errors, test_errors = get_all_errors(train_pct, x, y, st.session_state.seed, standardize, metric_choice)

    with col2:
        st.subheader("Fit Visualization (for selected degree)")
        steps_viz = [('poly', PolynomialFeatures(degree=st.session_state.poly_degree))]
        if standardize:
            steps_viz.append(('scaler', StandardScaler()))
        steps_viz.append(('model', LinearRegression()))
        pipeline_viz = Pipeline(steps_viz)
        pipeline_viz.fit(x_train.reshape(-1, 1), y_train)
        x_range = np.linspace(0, 1, 100).reshape(-1, 1)
        y_pred_range = pipeline_viz.predict(x_range)
        
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=x_train_s, y=y_train_s, mode='markers', name='Training Data', marker=dict(color='blue', opacity=0.7)))
        fig_fit.add_trace(go.Scatter(x=x_test_s, y=y_test_s, mode='markers', name='Test Data', marker=dict(color='red', opacity=0.7)))
        fig_fit.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='True Function (sin(2πx))', line=dict(color='green', dash='dash')))
        fig_fit.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred_range, mode='lines', name=f'Model (Degree {st.session_state.poly_degree})', line=dict(color='orange', width=3)))
        fig_fit.update_layout(title="Model Fit vs. Data", xaxis_title="x", yaxis_title="y", yaxis_range=[-2.5, 2.5], height=400)
        st.plotly_chart(fig_fit, use_container_width=True)

        st.subheader(f"Error ({metric_choice}) vs. Polynomial Degree")
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=degrees, y=train_errors, mode='lines+markers', name='Train Error'))
        fig_err.add_trace(go.Scatter(x=degrees, y=test_errors, mode='lines+markers', name='Test Error'))
        min_test_err_idx = np.argmin(test_errors)
        min_test_err_deg = degrees[min_test_err_idx]
        fig_err.add_annotation(x=min_test_err_deg, y=test_errors[min_test_err_idx], text=f"Min Test Error (Degree {min_test_err_deg})", showarrow=True, arrowhead=1, ax=20, ay=-40)
        fig_err.add_vline(x=min_test_err_deg, line=dict(color='red', dash='dot'))
        fig_err.update_layout(title="Train and Test Error vs. Model Complexity", xaxis_title="Polynomial Degree", yaxis_title=f"Error ({metric_choice})", height=400)
        st.plotly_chart(fig_err, use_container_width=True)

# ==============================================================================
# --- Tab 3: Logistic Regression ---
# ==============================================================================
with tab3:
    st.header("Tab 3: Logistic Regression")
    st.markdown("Adjust the threshold to see the trade-off between precision and recall.")

    X, y = get_logistic_data(st.session_state.n_samples, st.session_state.noise, st.session_state.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=st.session_state.seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        C = st.select_slider("Regularization Parameter (C)", options=[0.001, 0.01, 0.1, 1, 10, 100, 1000], value=1.0, help="Smaller C = stronger regularization.")
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        
        model = LogisticRegression(C=C, random_state=st.session_state.seed, solver='liblinear')
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        st.subheader("Metrics (at current threshold)")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}", help="Overall 'correctness'. Can be misleading if classes are imbalanced.")
        m_col2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.3f}", help="Area Under the Curve. 1.0 is perfect, 0.5 is random. Threshold-independent.")
        m_col3, m_col4 = st.columns(2)
        m_col3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}", help="Of '1' predictions, how many were correct? (TP / (TP + FP))")
        m_col4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}", help="Of all actual '1's, how many did we find? (TP / (TP + FN))")
        st.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}", help="Harmonic mean of Precision and Recall.")

        st.info("A **higher** threshold increases Precision but decreases Recall. A **lower** threshold does the opposite.")
        
        st.subheader("Model Coefficients")
        st.write(f"Feature 1 Coef: {model.coef_[0][0]:.3f}")
        st.write(f"Feature 2 Coef: {model.coef_[0][1]:.3f}")
        st.write(f"Intercept: {model.intercept_[0]:.3f}")
        
    with col2:
        st.subheader("Decision Boundary and Test Data")
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        Z = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
        
        df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2'])
        df_test['Class'] = y_test.astype(str)
        
        fig_bound = px.scatter(df_test, x='Feature 1', y='Feature 2', color='Class', color_discrete_map={'0': 'blue', '1': 'red'})
        fig_bound.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z, name="Prob(y=1)", showscale=False, contours=dict(start=0, end=1, size=0.1), opacity=0.5, line_width=0))
        fig_bound.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z, showscale=False, contours_coloring='lines', contours=dict(start=threshold, end=threshold, size=0), line=dict(color='black', width=3, dash='dash'), name=f"Threshold ({threshold})"))
        fig_bound.update_layout(title="Decision Boundary", height=500)
        st.plotly_chart(fig_bound, use_container_width=True)

    st.subheader("Model Performance Curves")
    plot_col1, plot_col2, plot_col3 = st.columns(3)
    with plot_col1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc_score(y_test, y_proba):.3f})")
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='Gray', dash='dash'))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig_roc, use_container_width=True)
    with plot_col2:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig_pr = px.line(x=recall, y=precision, title="Precision-Recall Curve")
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=400)
        st.plotly_chart(fig_pr, use_container_width=True)
    with plot_col3:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'], color_continuous_scale='Blues')
        fig_cm.update_layout(title="Confusion Matrix (at threshold)", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("**How to Read the Confusion Matrix:** Top-Left (True Negative), Bottom-Right (True Positive), Top-Right (False Positive), Bottom-Left (False Negative).")

# ==============================================================================
# --- Tab 4: Time Series (Data Leaks) ---
# ==============================================================================
with tab4:
    st.header("Tab 4: Time Series & Data Leaks")
    st.markdown("This tab demonstrates **data leaks**—using information from the future to train your model. The most common leak is **shuffling data** before splitting.")

    df_ts = get_timeseries_data(n_samples=300, noise=st.session_state.noise, seed=st.session_state.seed)
    
    st.subheader("1. Illustrative Time Series Data")
    st.markdown("Our synthetic data, with a trend, seasonality, and noise.")
    st.plotly_chart(px.line(df_ts, y='y', title='Synthetic Time Series Data'), use_container_width=True)
    
    st.subheader("2. Mathematical Foundations")
    st.markdown("A time series $Y_t$ is often a combination of Trend ($T_t$), Seasonality ($S_t$), and Noise ($\epsilon_t$). Models can be **Additive** ($Y_t = T_t + S_t + \epsilon_t$) or **Multiplicative** ($Y_t = T_t \times S_t \times \epsilon_t$). The key concept is **Autocorrelation**: the idea that $y_{t-1}$ is useful for predicting $y_t$.")
    
    st.subheader("3. Comparison of Time Series Models")
    st.markdown("""
| Model | Core Idea | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **ARIMA** | Statistical model using past values (AR), errors (MA), and differencing (I). | Interpretable, statistically robust. | Struggles with non-linear patterns. |
| **Neural Networks (RNN/LSTM)** | Sequential models with "memory" cells. | State-of-the-art for complex, non-linear data. | Black box, data-hungry, slow to train. |
| **Decision Trees (RF/GBM)** | Uses lagged values as features to make splits. | Good at non-linear interactions. | **CANNOT EXTRAPOLATE** (will never predict a value outside its training range). |
    """)

    st.subheader("4. Performance Comparison: Data Leak vs. Correct Method")
    n_lags = st.slider("Number of Lagged Features (Tab 4)", 1, 10, 3, key="tab4_lags")
    split_pct = st.slider("Train/Test Split (Time-Based) (Tab 4)", 0.6, 0.9, 0.7, 0.05, key="tab4_split")
    
    df_lagged = create_lagged_features(df_ts.copy(), lags=n_lags)
    X_features = df_lagged.drop(['y', 'time'], axis=1)
    y_labels = df_lagged['y']
    
    col_leak, col_correct = st.columns(2)
    with col_leak:
        st.error("The Wrong Way: Random Split (Data Leak)")
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_features, y_labels, test_size=0.3, random_state=st.session_state.seed, shuffle=True)
        model_l = LinearRegression()
        model_l.fit(X_train_l, y_train_l)
        pred_l = model_l.predict(X_test_l)
        rmse_l = np.sqrt(mean_squared_error(y_test_l, pred_l))
        st.metric("Deceptive RMSE (Random Split)", f"{rmse_l:.3f}")
        st.markdown("This RMSE is **deceptively low** because the model trained on data from the future.")
    with col_correct:
        st.success("The Correct Way: Time-Based Split")
        split_point = int(len(X_features) * split_pct)
        X_train_c, y_train_c = X_features.iloc[:split_point], y_labels.iloc[:split_point]
        X_test_c, y_test_c = X_features.iloc[split_point:], y_labels.iloc[split_point:]
        model_c = LinearRegression()
        model_c.fit(X_train_c, y_train_c)
        pred_c = model_c.predict(X_test_c)
        rmse_c = np.sqrt(mean_squared_error(y_test_c, pred_c))
        st.metric("Realistic RMSE (Time Split)", f"{rmse_c:.3f}")
        st.markdown("This is the **true performance**, as the model was only trained on the past.")

    st.subheader("5. Visualization of the Correct Model")
    df_train_plot = df_lagged.iloc[:split_point]
    df_test_plot = df_lagged.iloc[split_point:].copy()
    df_test_plot['prediction'] = pred_c
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_train_plot.index, y=df_train_plot['y'], name='Train Data', line=dict(color='blue')))
    fig_ts.add_trace(go.Scatter(x=df_test_plot.index, y=df_test_plot['y'], name='Test Data (Actual)', line=dict(color='orange')))
    fig_ts.add_trace(go.Scatter(x=df_test_plot.index, y=df_test_plot['prediction'], name='Model Prediction', line=dict(color='red', dash='dash')))
    
    split_date = df_lagged.index[split_point]
    fig_ts.add_vline(x=split_date, line=dict(color='gray', dash='dot'))
    fig_ts.add_annotation(x=split_date, y=1.0, yref="paper", text="Train/Test Split", showarrow=False, xanchor="left", yanchor="top", xshift=5)
    
    fig_ts.update_layout(title="Time Series Forecast (Correct Method)", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig_ts, use_container_width=True)

# ==============================================================================
# --- NEW: Tab 5: Time Series Model Playground (No pmdarima) ---
# ==============================================================================
# Helper Functions for Tab 5
@st.cache_data
def get_playground_data(dataset_name):
    if dataset_name == "Air Passengers (Real)":
        # Load from statsmodels
        data_df = sm.datasets.airpassengers.load_pandas().data
        data = data_df['value']
        data.index = pd.to_datetime(data_df['date'])
        data.name = 'y'
    elif dataset_name == "Stationary (No Trend)":
        time_idx = np.arange(200)
        seasonality = np.sin(2 * np.pi * time_idx / 12) * 20
        noise = np.random.randn(200) * 5
        data = pd.Series(seasonality + noise, name="y")
        data.index = pd.date_range(start="2010-01-01", periods=200, freq='MS')
    else: # "Trend + Seasonality (Synthetic)"
        df = get_timeseries_data(n_samples=200, noise=0.8, seed=42)
        data = df['y']
    return data

# REPLACED auto_arima with SARIMAX
@st.cache_data
def fit_sarimax(data, seasonal_period):
    start_time = time.time()
    # We're hardcoding a common SARIMA order (p,d,q) (P,D,Q,m)
    # This is a reasonable default for demonstration
    model = SARIMAX(
        data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # Use .fit(disp=False) to hide convergence logs
    results = model.fit(disp=False)
    fit_time = time.time() - start_time
    return results, fit_time

def prepare_ml_data(series, n_lags):
    df = pd.DataFrame(series.copy())
    df.columns = ['y'] # Ensure column is 'y'
    df['time_index'] = np.arange(len(df))
    for i in range(1, n_lags + 1):
        df[f'y_lag_{i}'] = df['y'].shift(i)
    df = df.dropna()
    
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y

@st.cache_data
def fit_ml_model(X_train, y_train, model_type):
    if model_type == "Linear Regression":
        model = LinearRegression()
    else: # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=st.session_state.seed)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    return model, fit_time

def forecast_ml_model(model, X_train, y_train, n_lags, horizon):
    last_row = X_train.iloc[[-1]].copy()
    last_y = y_train.iloc[-1]
    
    forecasts = []
    
    for i in range(horizon):
        last_row['time_index'] += 1
        if n_lags > 0:
            for j in range(n_lags, 1, -1):
                last_row[f'y_lag_{j}'] = last_row[f'y_lag_{j-1}']
            last_row['y_lag_1'] = last_y
            
        next_pred = model.predict(last_row)[0]
        forecasts.append(next_pred)
        last_y = next_pred
        
    return np.array(forecasts)

with tab5:
    st.header("Tab 5: Time Series Model Playground")
    st.markdown("Compare different forecasting models on a dataset. **Note:** `SARIMAX` can be slow the first time it runs.")
    
    # --- 1. Controls ---
    st.subheader("1. Controls")
    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox("Select a Template Dataset", ["Trend + Seasonality (Synthetic)", "Air Passengers (Real)", "Stationary (No Trend)"])
        train_split = st.slider("Train Split", 0.6, 0.9, 0.8, 0.05)
    with col2:
        # Guess seasonal period (m)
        seasonal_period = 12 if "Air" in dataset else 50 if "Trend" in dataset else 12
        n_lags_ml = st.slider("Lags for ML Models", 1, 15, seasonal_period)
        ci_alpha = st.slider("Confidence Interval", 0.01, 0.5, 0.05, 0.01, help="e.g., 0.05 = 95% CI")

    # --- 2. Load and Split Data ---
    data = get_playground_data(dataset)
    split_point = int(len(data) * train_split)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    n_test = len(test_data)
    n_forecast = n_test + 12
    
    st.subheader("2. Dataset and Split")
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(x=train_data.index, y=train_data, name="Train Data", line=dict(color='blue')))
    fig_data.add_trace(go.Scatter(x=test_data.index, y=test_data, name="Test Data (Actual)", line=dict(color='orange')))
    # Get the split date
    split_date = train_data.index[-1]
    
    # Step 1: Add the line
    fig_data.add_vline(x=split_date, line=dict(color='gray', dash='dot'))
    
    # Step 2: Add the annotation manually
    fig_data.add_annotation(
        x=split_date,
        y=1.0,                   # Position at the top
        yref="paper",            # Use plot's y-axis percentage
        text="Train/Test Split",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        xshift=5                 # Nudge it 5px to the right
    )
    fig_data.update_layout(title=f"Data: {dataset}", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig_data, use_container_width=True)

    # --- 3. Run Models ---
    st.subheader("3. Model Fitting and Comparison")
    
    with st.spinner(f"Fitting 3 models... `SARIMAX` may take a moment."):
        # Model 1: SARIMAX (replaces auto_arima)
        sarimax_model, arima_time = fit_sarimax(train_data, seasonal_period)
        
        # Create forecast index
        forecast_index = pd.date_range(start=test_data.index[0], periods=n_forecast, freq=train_data.index.freqstr)
        
        # Get predictions
        pred_obj = sarimax_model.get_prediction(start=test_data.index[0], end=forecast_index[-1])
        arima_preds = pred_obj.predicted_mean
        
        # Get confidence intervals
        ci_obj = pred_obj.conf_int(alpha=ci_alpha)
        # Rename columns to be generic 'lower' and 'upper'
        arima_ci_df = pd.DataFrame(ci_obj).rename(columns={f'lower {data.name}': 'lower', f'upper {data.name}': 'upper'})
        
        # Model 2 & 3: ML Models
        X, y = prepare_ml_data(data, n_lags_ml)
        X_train, y_train = X.iloc[:(split_point - n_lags_ml)], y.iloc[:(split_point - n_lags_ml)]
        
        # Linear Regression
        lr_model, lr_time = fit_ml_model(X_train, y_train, "Linear Regression")
        lr_preds = forecast_ml_model(lr_model, X_train, y_train, n_lags_ml, n_forecast)
        
        # Random Forest
        rf_model, rf_time = fit_ml_model(X_train, y_train, "Random Forest")
        rf_preds = forecast_ml_model(rf_model, X_train, y_train, n_lags_ml, n_forecast)

    # --- 4. Compare Results ---
    metrics = {
        "Model": ["SARIMAX", "Linear Regression", "Random Forest"],
        "RMSE (Test Set)": [
            np.sqrt(mean_squared_error(test_data, arima_preds[:n_test])),
            np.sqrt(mean_squared_error(test_data, lr_preds[:n_test])),
            np.sqrt(mean_squared_error(test_data, rf_preds[:n_test]))
        ],
        "Fit Time (s)": [f"{arima_time:.2f}", f"{lr_time:.2f}", f"{rf_time:.2f}"]
    }
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    
    st.info(
        """
        **How to Read This:**
        * **SARIMAX:** Statistical model. Only model that provides a true, statistical **Confidence Interval**.
        * **Linear Regression:** ML model. Can extrapolate the `time_index` trend, so its forecast continues the trend.
        * **Random Forest:** ML model. **Cannot extrapolate!** Notice its forecast flatlines. It can only predict values within the range it was trained on.
        """
    )
    
    st.subheader("4. Forecast Visualization")
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=train_data.index, y=train_data, name="Train Data", line=dict(color='blue', width=2)))
    fig_all.add_trace(go.Scatter(x=test_data.index, y=test_data, name="Test Data (Actual)", line=dict(color='orange', width=2)))
    
    # SARIMAX
    fig_all.add_trace(go.Scatter(x=forecast_index, y=arima_preds, name="SARIMAX Forecast", line=dict(color='green', dash='dash')))
    fig_all.add_trace(go.Scatter(x=forecast_index, y=arima_ci_df['lower'], name=f"SARIMAX {(1-ci_alpha)*100}% CI", line=dict(color='green', width=0.5), fill=None))
    fig_all.add_trace(go.Scatter(x=forecast_index, y=arima_ci_df['upper'], name="SARIMAX CI Upper", line=dict(color='green', width=0.5), fill='tonexty', fillcolor='rgba(0,176,80,0.2)'))
    
    # Linear Regression
    fig_all.add_trace(go.Scatter(x=forecast_index, y=lr_preds, name="Linear Regression Forecast", line=dict(color='red', dash='dash')))
    
    # Random Forest
    fig_all.add_trace(go.Scatter(x=forecast_index, y=rf_preds, name="Random Forest Forecast", line=dict(color='purple', dash='dash')))

    fig_all.update_layout(title="Model Forecast Comparison", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig_all, use_container_width=True)
    
    # --- 5. Deep Dives ---
    st.subheader("5. Model Deep Dives")
    tab_arima, tab_lr, tab_rf = st.tabs(["SARIMAX Details", "Linear Regression Details", "Random Forest Details"])
    
    with tab_arima:
        st.markdown("We fit a `SARIMAX(1,1,1)(1,1,1,m)` model from `statsmodels`.")
        st.markdown(f"This is a Seasonal ARIMA model with `m={seasonal_period}`.")
        st.text(sarimax_model.summary())
        
    with tab_lr:
        st.markdown("Linear Regression creates features from the lags and a `time_index` (a simple count: 0, 1, 2...).")
        st.markdown("Because it learns a coefficient for `time_index`, it can **extrapolate** a linear trend into the future.")
        st.write("Model Coefficients:")
        st.dataframe(pd.DataFrame({'feature': X_train.columns, 'coefficient': lr_model.coef_}))
        
    with tab_rf:
        st.markdown("Random Forest also uses lags and the `time_index`.")
        st.warning(
            """
            **Key Teaching Point:** Notice how the Random Forest forecast flatlines?
            
            This is because tree-based models **cannot extrapolate**. They can only
            predict by averaging values they have seen during training. Once the `time_index`
            feature goes beyond its training range, the model defaults to predicting the
            average of the highest values it ever saw. This makes it a poor choice for
            datasets with a strong, continuing trend.
            """
        )
        st.write("Feature Importances:")
        st.dataframe(pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_}))