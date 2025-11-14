import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time  # To time model fits
import matplotlib.pyplot as plt # For decision tree plotting

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor # FOR NEW ANN PAGE
from sklearn.exceptions import ConvergenceWarning # FOR NEW ANN PAGE
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
from sklearn.tree import DecisionTreeClassifier, plot_tree # For Tab 6

# --- Page Configuration ---
st.set_page_config(
    page_title="Understanding Regression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.header("Global Controls")
    
    # Use session state for persistent controls
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

    # --- Page navigation ---
    st.markdown("---")
    st.header("Navigation")
    page_options = [
        "Gradient Descent",
        "Manual Linear Fit", # This page is now updated
        "Train vs Test (Overfitting)", 
        "Logistic Regression",
        "Decision Trees",
        "Neural Network Training", # Replaced "Manual Fit"
        "Time Series (Data Leaks)", 
        "TS Analysis"
    ]
    page = st.radio("Select a topic", page_options, label_visibility="collapsed")
    st.markdown("---")

    st.header("Key Formulas")
    st.latex(r"\hat{y}_i = \beta_0 + \beta_1 x_i")
    st.latex(r"R^2 = 1 - \frac{\text{SSE}}{\text{SST}}")
    st.latex(r"p(y=1|x)=\sigma(\beta_0+\beta^\top x)")
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
    st.latex(r"\text{Gini} = 1 - \sum_{i=1}^{C} (p_i)^2")
    
    st.markdown("---")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")


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

# Time Series Data Function (Tab 4 & 5)
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

# Helper function for Tab 3 and Tab 6
@st.cache_data
def get_manufacturing_data(seed):
    np.random.seed(seed)
    n_samples = 25
    temp = np.random.normal(loc=100, scale=10, size=n_samples).round(1)
    pressure = np.random.normal(loc=50, scale=5, size=n_samples).round(1)
    last_fail = np.random.randint(0, 2, size=n_samples)
    
    score = -15 + (temp * 0.1) + (pressure * 0.05) + (last_fail * 2.5)
    prob = 1 / (1 + np.exp(-score))
    y = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'temperature': temp,
        'pressure': pressure,
        'last_year_failed': last_fail,
        'failed_this_year': y
    })
    return df

# Helper function for Tab 5
@st.cache_data
def get_playground_data(dataset_name, n_samples, noise, seed):
    if dataset_name == "Air Passengers (Real)":
        time_idx = np.arange(144)
        trend = np.exp(time_idx * 0.05)
        seasonality = np.sin(2 * np.pi * time_idx / 12) * 20 + 50
        y = (trend + seasonality + np.random.randn(144) * 10).astype(int)
        data = pd.Series(y, name="y")
        data.index = pd.date_range(start="1949-01-01", periods=144, freq='MS')
    elif dataset_name == "Stationary (No Trend)":
        time_idx = np.arange(n_samples)
        seasonality = np.sin(2 * np.pi * time_idx / 12) * 20
        noise_val = np.random.randn(n_samples) * 5
        data = pd.Series(seasonality + noise_val, name="y")
        data.index = pd.date_range(start="2010-01-01", periods=n_samples, freq='MS')
    else: # "Trend + Seasonality (Synthetic)"
        df = get_timeseries_data(n_samples=n_samples, noise=noise, seed=seed)
        data = df['y']
    return data

# Helper function for Tab 7 (ANN)
@st.cache_data
def get_nn_data(seed, noise):
    np.random.seed(seed)
    x = np.linspace(0, 1, 10).reshape(-1, 1) # 10 points, as 2D array
    y_true = np.sin(2 * np.pi * x.ravel())
    y = y_true + np.random.randn(10) * noise
    return x, y, y_true


# --- Main App ---
st.title("Understanding Supervised Learning")

# ==============================================================================
# --- Page 1: Gradient Descent ---
# ==============================================================================
if page == "Gradient Descent":
    st.header("Gradient Descent in a Sandpit")
    st.markdown(
        """
        This page demonstrates **Gradient Descent**, the core optimization algorithm
        used to train (or "fit") most machine learning models.
        
        The goal is to find the **lowest point** in a "sandpit" (the cost function).
        We do this by taking small steps in the direction of the **steepest slope** (the negative gradient).
        """
    )

    # --- 1. Define the Function and its Gradient ---
    def cost_function(x, y):
        return x**2 + 2*y**2

    def gradient(x, y):
        grad_x = 2 * x
        grad_y = 4 * y
        return grad_x, grad_y

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        
        start_x = st.slider("Start Position (x)", -10.0, 10.0, 8.0)
        start_y = st.slider("Start Position (y)", -10.0, 10.0, 6.0)
        
        learning_rate = st.slider("Learning Rate (γ)", 0.01, 1.0, 0.1, 0.01,
                                  help="How big of a step to take. Try a large value (e.g., > 0.5) to see it fail!")
        
        n_steps = st.slider("Number of Steps", 10, 100, 30)

        st.info(
            """
            **Try This:**
            * **High Learning Rate (e.g., 0.6):** Watch the point "overshoot" the minimum and diverge!
            * **Low Learning Rate (e.g., 0.01):** Watch how slowly it converges.
            * **Start Position:** See how the path always moves perpendicular to the contour lines.
            """
        )

    # --- 2. Run Gradient Descent ---
    path = []
    current_x, current_y = start_x, start_y
    
    for i in range(n_steps):
        path.append((current_x, current_y))
        grad_x, grad_y = gradient(current_x, current_y)
        current_x = current_x - learning_rate * grad_x
        current_y = current_y - learning_rate * grad_y
        
    path = np.array(path)
    path_z = cost_function(path[:, 0], path[:, 1])

    # --- 3. Create Plots ---
    with col2:
        # Create 3D Surface
        x_grid = np.linspace(-10, 10, 50)
        y_grid = np.linspace(-10, 10, 50)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        Z_mesh = cost_function(X_mesh, Y_mesh)

        fig_3d = go.Figure(data=[
            go.Surface(z=Z_mesh, x=X_mesh, y=Y_mesh, opacity=0.7, colorscale='viridis')
        ])
        fig_3d.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path_z,
            mode='lines+markers',
            name='Gradient Path',
            line=dict(color='red', width=4),
            marker=dict(size=4)
        ))
        fig_3d.update_layout(title="3D View: The 'Sandpit' (Cost Surface)", height=400, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

        # Create 2D Contour Plot
        fig_2d = go.Figure(data=[
            go.Contour(z=Z_mesh, x=x_grid, y=y_grid, colorscale='viridis', ncontours=20)
        ])
        fig_2d.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1],
            mode='lines+markers',
            name='Gradient Path',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
        fig_2d.update_layout(title="2D View: Path to the Minimum", height=400, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_2d, use_container_width=True)


# ==============================================================================
# --- Page 2: Manual Linear Fit (NOW WITH COST SURFACE) ---
# ==============================================================================
elif page == "Manual Linear Fit":
    st.header("Manual Linear Fit")
    st.markdown(
        """
        Try to minimize SSE manually. This page connects to Gradient Descent: the plot on the
        right is the 'cost surface' or 'sandpit' for our model. 
        
        **Your goal is to find the bottom of the blue valley by moving the sliders.**
        """
    )
    
    # --- This function calculates the cost surface ---
    @st.cache_data
    def calculate_sse_surface(x_data, y_data):
        # Create a grid of parameter values (b0, b1)
        b0_vals = np.linspace(-2, 6, 40)  
        b1_vals = np.linspace(0, 6, 40)  
        sse_grid = np.zeros((40, 40))
        
        # Calculate SSE for each (b0, b1) pair
        for i, b0 in enumerate(b0_vals):
            for j, b1 in enumerate(b1_vals):
                y_pred = b0 + b1 * x_data
                sse = np.sum((y_data - y_pred)**2)
                sse_grid[i, j] = sse
                
        return b0_vals, b1_vals, sse_grid
    # --- End of new function ---

    x, y, true_b0, true_b1 = get_linear_data(st.session_state.n_samples, st.session_state.noise, st.session_state.seed)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        b0 = st.slider("Intercept (β₀)", -2.0, 6.0, 0.0, 0.1) 
        b1 = st.slider("Slope (β₁)", 0.0, 6.0, 1.0, 0.1)     
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
        # --- PLOT 1: The original scatter plot ---
        st.subheader("Model Space (Data)")
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
        
        fig.update_layout(title="Manual Linear Regression Fit", xaxis_title="x", yaxis_title="y", height=400, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)

        # --- PLOT 2: The new cost surface plot ---
        st.subheader("Parameter Space (Cost Surface)")
        b0_vals, b1_vals, sse_grid = calculate_sse_surface(x, y)
        
        fig_cost = go.Figure(data=[
            go.Contour(
                z=sse_grid, 
                x=b1_vals,  # x-axis is Slope (b1)
                y=b0_vals,  # y-axis is Intercept (b0)
                colorscale='viridis_r', # blue is low, yellow is high
                ncontours=30,
                name='SSE Surface'
            )
        ])
        fig_cost.add_trace(go.Scatter(
            x=[b1], y=[b0],
            mode='markers',
            name='Your Manual Fit',
            marker=dict(color='red', size=12, symbol='x')
        ))
        if 'ols_b0' in st.session_state:
            fig_cost.add_trace(go.Scatter(
                x=[st.session_state.ols_b1], y=[st.session_state.ols_b0],
                mode='markers',
                name='Optimal OLS Fit (The Minimum)',
                marker=dict(color='cyan', size=12, symbol='star')
            ))
        fig_cost.update_layout(
            title="Cost Surface (SSE) vs. Parameters",
            xaxis_title="Slope (β₁)",
            yaxis_title="Intercept (β₀)",
            height=400,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_cost, use_container_width=True)

# ==============================================================================
# --- Page 3: Training vs Testing: Overfitting ---
# ==============================================================================
elif page == "Train vs Test (Overfitting)":
    st.header("Training vs Testing: Overfitting")
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
        # --- FIX: Was height=4.00 ---
        fig_err.update_layout(title="Train and Test Error vs. Model Complexity", xaxis_title="Polynomial Degree", yaxis_title=f"Error ({metric_choice})", height=400)
        st.plotly_chart(fig_err, use_container_width=True)

# ==============================================================================
# --- Page 4: Logistic Regression ---
# ==============================================================================
elif page == "Logistic Regression":
    st.header("Logistic Regression")
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

    st.subheader("Context: Manufacturing Failure Example")
    with st.expander("Click to see a conceptual explanation and manufacturing example"):
        st.markdown(
            """
            #### Why Logistic Regression?
            Our target variable (e.g., `failed_this_year`) is **binary** (0 or 1). A normal Linear Regression model $y = \beta_0 + \beta_1 x$ is unbounded and can predict values like -0.5 or 1.5, which are meaningless as probabilities.
            We need a function that maps any real number to the range [0, 1]. We use the **logistic (or sigmoid) function**, which has an "S" shape.
            $$ \text{Probability} = \sigma(z) = \frac{1}{1 + e^{-z}} $$
            Where $z$ is our familiar linear equation: $z = \beta_0 + \beta_1 x_1 + ...$
            This $z$ value is called the **log-odds**. The coefficients ($\beta_1$, etc.) are no longer interpreted directly. Instead, we interpret their exponent, $e^\beta$, which is the **Odds Ratio**.
            """
        )
        st.subheader("Example: Predicting Machine Failure")
        st.markdown("Let's predict `failed_this_year` (0 or 1) based on `temperature`, `pressure`, and `last_year_failed`.")
        
        df_mfg = get_manufacturing_data(st.session_state.seed)
        X_mfg = df_mfg.drop('failed_this_year', axis=1)
        y_mfg = df_mfg['failed_this_year']
        
        mfg_model = LogisticRegression()
        mfg_model.fit(X_mfg, y_mfg)
        
        st.dataframe(df_mfg, use_container_width=True)
        
        st.subheader("Model Interpretation: Coefficients & Odds Ratios")
        features = X_mfg.columns
        coeffs = mfg_model.coef_[0]
        odds_ratios = np.exp(coeffs)
        
        interp_df = pd.DataFrame({
            'Feature': features,
            'Coefficient (Log-Odds)': coeffs,
            'Odds Ratio (e^coeff)': odds_ratios
        })
        st.dataframe(interp_df.round(3), use_container_width=True)
        st.markdown(
            """
            **How to Interpret:**
            * **Odds Ratio:** An odds ratio of **1.15** means a 1-unit increase in that feature (e.g., 1 degree of temperature) makes the event **15% more likely**. An odds ratio of **0.80** makes it **20% less likely**.
            From our model, a 1-degree rise in `temperature` increases the odds of failure by ~10%, while a machine that `last_year_failed` is over 10x more likely to fail again!
            """
        )

# ==============================================================================
# --- Page 5: Decision Trees ---
# ==============================================================================
elif page == "Decision Trees":
    st.header("Decision Tree Classification")
    st.markdown(
        """
        Unlike Logistic Regression (which finds a single linear boundary), a **Decision Tree**
        asks a series of simple, axis-aligned questions to split the data into "pure"
        groups (e.g., all "Fail" or all "Succeed").
        
        It builds this tree by finding the best question to ask at each step. "Best" is
        measured by how much the question reduces **impurity**.
        """
    )
    
    st.subheader("1. Key Concepts: Gini vs. Entropy")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            """
            #### Gini Impurity
            * **Formula:** $G = 1 - \sum_{i=1}^{C} (p_i)^2$
            * **What it is:** The probability of *incorrectly* classifying a randomly chosen
                item if it were randomly labeled according to the class distribution.
            * **Range:** 0 (perfectly pure) to 0.5 (perfectly mixed).
            """
        )
    with col2:
        st.info(
            """
            #### Entropy
            * **Formula:** $H = - \sum_{i=1}^{C} p_i \log_2(p_i)$
            * **What it is:** A measure of "disorder." A node with
                high entropy is very mixed.
            * **Range:** 0 (perfectly pure) to 1 (perfectly mixed).
            """
        )

    st.subheader("2. Interactive Model (Manufacturing Example)")
    st.markdown("Here is the *same* manufacturing dataset. Change the controls to see how the tree's structure changes.")
    
    col1, col2 = st.columns(2)
    with col1:
        df_mfg = get_manufacturing_data(st.session_state.seed)
        X_mfg = df_mfg.drop('failed_this_year', axis=1)
        y_mfg = df_mfg['failed_this_year']
        
        criterion = st.selectbox("Impurity Criterion", ["gini", "entropy"])
        max_depth = st.slider("Tree Max Depth", min_value=1, max_value=5, value=3)
        
        st.dataframe(df_mfg, height=300)

    with col2:
        tree_model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=st.session_state.seed
        )
        tree_model.fit(X_mfg, y_mfg)
        
        st.markdown("#### Visualized Decision Tree")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(
            tree_model,
            ax=ax,
            feature_names=X_mfg.columns.tolist(),
            class_names=["Succeed", "Fail"], # 0 = Succeed, 1 = Fail
            filled=True,
            rounded=True,
            fontsize=10
        )
        st.pyplot(fig)
        plt.close(fig) # Clear figure from memory
    
    st.info(
        """
        **How to Read This Tree:**
        * **Top Node (Root):** The first question the tree asks.
        * **Gini/Entropy:** The impurity of the node (goal is 0).
        * **Samples:** How many data points are in this node.
        * **Value:** `[# of class 0, # of class 1]`. Here, `[Succeed, Fail]`.
        * **Class:** The majority class in that node.
        * **Leaves (Bottom Nodes):** The final predictions.
        """
    )

# ==============================================================================
# --- Page 6: Neural Network Training (REPLACES Manual Fit) ---
# ==============================================================================
elif page == "Neural Network Training":
    st.header("Neural Network Training (Backpropagation)")
    st.markdown(
        """
        This page demonstrates **Backpropagation** (Gradient Descent for networks).
        We'll use `sklearn`'s `MLPRegressor` to find the *actual* best weights
        and biases to fit the 10-point sine wave from your `Backpropagation.ipynb` notebook.
        
        Configure the network's architecture, then press **"Train Model"** to see it learn.
        """
    )

    # Get data
    x_data, y_data, y_true = get_nn_data(st.session_state.seed, st.session_state.noise)
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1) # For a smooth true line

    # --- 1. Architecture Controls ---
    st.subheader("1. Network Architecture")
    st.markdown("Define the structure of your neural network.")
    
    n_layers = st.slider("Number of Hidden Layers", 1, 4, 1)
    
    hidden_neurons_list = []
    cols = st.columns(n_layers)
    for i in range(n_layers):
        with cols[i]:
            neurons = st.slider(f"Neurons in Layer {i+1}", 1, 50, 10, key=f"layer_{i}_neurons")
            hidden_neurons_list.append(neurons)
    hidden_layer_sizes = tuple(hidden_neurons_list) # e.g., (10,) or (20, 10)

    # --- 2. Parameter Calculation ---
    st.subheader("2. Model Complexity")
    
    def calculate_total_parameters(layer_sizes):
        total_params = 0
        for i in range(1, len(layer_sizes)):
            weights = layer_sizes[i-1] * layer_sizes[i]
            biases = layer_sizes[i]
            total_params += (weights + biases)
        return total_params

    # [1] input, *hidden_layer_sizes, [1] output
    full_layer_list = [1] + list(hidden_layer_sizes) + [1] 
    total_params = calculate_total_parameters(full_layer_list)
    
    st.metric("Total Trainable Parameters (Weights & Biases)", f"{total_params}")
    st.markdown("More parameters make the model more 'flexible' but also slower and more prone to overfitting.")

    # --- 3. Training Controls ---
    st.subheader("3. Training Controls")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.select_slider("Learning Rate", 
                                         options=[0.0001, 0.001, 0.01, 0.1, 1.0], 
                                         value=0.01)
    with col2:
        n_epochs = st.number_input("Total Epochs", 100, 10000, 2000)

    # --- 4. Training ---
    st.subheader("4. Training Animation")
    
    if st.button("Train Model"):
        col1, col2 = st.columns(2)
        with col1:
            plot_placeholder = st.empty()
        with col2:
            loss_placeholder = st.empty()

        # Create the model
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, # Use dynamic sizes
            activation='relu',
            solver='sgd',
            learning_rate_init=learning_rate,
            max_iter=1, # We control iterations manually
            warm_start=True, # Allow retraining
            random_state=st.session_state.seed
        )
        
        loss_history = []
        n_batches = 50 # Number of animation frames
        epochs_per_batch = max(1, n_epochs // n_batches)
        
        with st.spinner("Training model..."):
            st.warning("Training in progress... (Convergence warnings are normal!)", icon="🤖")
            
            for i in range(n_batches):
                try:
                    model.fit(x_data, y_data)
                except ConvergenceWarning:
                    pass # Ignore warnings during partial fits
                
                loss = model.loss_
                loss_history.append(loss)
                
                # Update plots
                y_pred_plot = model.predict(x_plot)
                
                # Plot 1: Model Fit
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=x_data.ravel(), y=y_data, mode='markers', name='Data Points (Real)', marker=dict(size=10)))
                fig_fit.add_trace(go.Scatter(x=x_plot.ravel(), y=np.sin(2 * np.pi * x_plot.ravel()), mode='lines', name='True Sine Wave', line=dict(dash='dash', color='green')))
                fig_fit.add_trace(go.Scatter(x=x_plot.ravel(), y=y_pred_plot, mode='lines', name='ANN Forecast', line=dict(color='orange', width=3)))
                fig_fit.update_layout(title=f"Model Fit (Epoch {(i+1) * epochs_per_batch})", xaxis_title="x", yaxis_title="y", yaxis_range=[-2.5, 2.5], margin=dict(l=0, r=0, b=0, t=40))
                plot_placeholder.plotly_chart(fig_fit, use_container_width=True)
                
                # Plot 2: Loss Curve
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=np.arange(len(loss_history)), y=loss_history, mode='lines', name='Loss'))
                fig_loss.update_layout(title="Training Loss (Cost)", xaxis_title="Training Batch", yaxis_title="Loss", yaxis_type="log", margin=dict(l=0, r=0, b=0, t=40))
                loss_placeholder.plotly_chart(fig_loss, use_container_width=True)
                
                # Re-fit for next batch
                model.max_iter += epochs_per_batch
                
            st.success("Training complete!")
            
            # --- 5. Final Performance ---
            st.subheader("5. Model Performance")
            st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
            
            y_pred_data = model.predict(x_data)
            final_rmse = np.sqrt(mean_squared_error(y_data, y_pred_data))
            st.metric("Final Model RMSE (on 10 points)", f"{final_rmse:.4f}")

            st.subheader("Final Data & Forecasts")
            df_nn = pd.DataFrame({
                'x': x_data.ravel(),
                'y_real': y_data.round(2),
                'y_forecasted': y_pred_data.round(2)
            })
            st.dataframe(df_nn, use_container_width=True)
    
    else:
        st.info("Click 'Train Model' to start the backpropagation process.")


# ==============================================================================
# --- Page 7: Time Series (Data Leaks) ---
# ==============================================================================
elif page == "Time Series (Data Leaks)":
    st.header("Time Series & Data Leaks")
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
    n_lags = st.slider("Number of Lagged Features", 1, 10, 3, key="tab4_lags")
    split_pct = st.slider("Train/Test Split (Time-Based)", 0.6, 0.9, 0.7, 0.05, key="tab4_split")
    
    df_lagged = create_lagged_features(df_ts.copy(), lags=n_lags)
    X_features = df_lagged.drop(['y', 'time'], axis=1)
    y_labels = df_lagged['y'] # --- FIX: Was df_lagdged ---
    
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
# --- Page 8: Time Series Analysis (Fast Version) ---
# ==============================================================================
elif page == "TS Analysis":
    st.header("Time Series Analysis (Rolling Windows)")
    st.markdown(
        """
        This is a fast, simple analysis technique called a **rolling window**.
        It helps us visualize the **trend** and **volatility** of the data.
        
        * **Rolling Average:** Smooths out short-term noise to reveal the underlying trend.
        * **Rolling Standard Deviation:** Shows if the data's volatility (randomness) is constant or changing over time.
        """
    )
    
    st.subheader("1. Controls")
    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox(
            "Select a Template Dataset", 
            ["Trend + Seasonality (Synthetic)", "Air Passengers (Real)", "Stationary (No Trend)"]
        )
    with col2:
        window_size = st.slider("Rolling Window Size", min_value=2, max_value=50, value=12,
                                help="How many data points to 'average' over. A larger window means more smoothing.")

    data = get_playground_data(
        dataset, 
        n_samples=st.session_state.n_samples, 
        noise=st.session_state.noise, 
        seed=st.session_state.seed
    )
    
    st.subheader("2. Raw Data")
    fig_data = px.line(data, title=f"Data: {dataset}", labels={"value": "Value", "index": "Date"})
    st.plotly_chart(fig_data, use_container_width=True)

    st.subheader("3. Rolling Window Analysis")
    
    df = pd.DataFrame(data)
    df.columns = ['y'] 
    df['rolling_avg'] = df['y'].rolling(window=window_size).mean()
    df['rolling_std'] = df['y'].rolling(window=window_size).std()
    
    fig_rolling = go.Figure()
    
    fig_rolling.add_trace(go.Scatter(
        x=df.index, y=df['y'], name="Raw Data",
        line=dict(color='blue', width=1), opacity=0.5
    ))
    fig_rolling.add_trace(go.Scatter(
        x=df.index, y=df['rolling_avg'], name="Rolling Average (Trend)",
        line=dict(color='orange', width=3)
    ))
    fig_rolling.add_trace(go.Scatter(
        x=df.index, y=df['rolling_std'], name="Rolling Std. Dev. (Volatility)",
        line=dict(color='green', width=2, dash='dash'),
        yaxis="y2" 
    ))
    
    fig_rolling.update_layout(
        title=f"Rolling Window Analysis (Window Size = {window_size})",
        xaxis_title="Date",
        yaxis=dict(title="Value (Avg & Raw)"),
        yaxis2=dict(
            title="Standard Deviation",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0, y=1, traceorder="normal")
    )
    
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    st.info(
        """
        **How to Read This Chart:**
        * Change the **Rolling Window Size** slider. A larger window creates a smoother "Rolling Average" line.
        * **"Air Passengers"**: Notice how the "Rolling Std. Dev." (green line) *increases* over time? This means the volatility is growing.
        * **"Stationary (No Trend)"**: Notice how the "Rolling Average" stays flat (around 0) and the "Rolling Std. Dev." is also flat? This is the definition of a **stationary** series.
        """
    )