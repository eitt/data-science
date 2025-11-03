import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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
    true_b0 = 2
    true_b1 = 3
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
    # Using make_classification for 2 distinct, slightly overlapping clusters
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.01,
        class_sep=0.5 + (1.5-noise), # Noise slider controls class separation
        random_state=seed,
    )
    return X, y

# --- Main App ---
st.title("Understanding Regression: Linear, Polynomial, and Logistic")

tab1, tab2, tab3 = st.tabs(
    ["Manual Linear Fit", "Train vs Test (Overfitting)", "Logistic Regression"]
)

# ==============================================================================
# --- Tab 1: Manual Linear Fit ---
# ==============================================================================
with tab1:
    st.header("Tab 1: Manual Linear Fit")
    st.markdown(
        "Try to minimize SSE manually using the sliders. Then click 'Compute OLS' to compare your line with the optimal one."
    )
    
    x, y, true_b0, true_b1 = get_linear_data(
        st.session_state.n_samples, st.session_state.noise, st.session_state.seed
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        b0 = st.slider("Intercept (β₀)", -5.0, 5.0, 0.0, 0.1)
        b1 = st.slider("Slope (β₁)", -5.0, 5.0, 1.0, 0.1)
        
        show_true_line = st.checkbox("Show true underlying line", value=False)
        show_residuals = st.checkbox("Show vertical projections (residuals)", value=True)
        
        # Calculations
        y_pred = b0 + b1 * x
        residuals = y - y_pred
        sse = np.sum(residuals**2)
        sst = np.sum((y - y.mean())**2)
        
        # Avoid division by zero if SST is 0
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
        # Visualization
        fig = go.Figure()
        
        # Scatter plot of data
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data (x, y)'))
        
        # Manual fit line
        fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Manual Fit (ŷ)', line=dict(color='orange', width=2)))
        
        # True line
        if show_true_line:
            y_true_line = true_b0 + true_b1 * x
            fig.add_trace(go.Scatter(x=x, y=y_true_line, mode='lines', name='True Line', line=dict(color='blue', dash='dash', width=2)))
        
        # OLS line (if computed)
        if 'ols_b0' in st.session_state:
            y_ols_line = st.session_state.ols_b0 + st.session_state.ols_b1 * x
            fig.add_trace(go.Scatter(x=x, y=y_ols_line, mode='lines', name='OLS Fit', line=dict(color='green', dash='dot', width=3)))

        # Residuals
        if show_residuals:
            for i in range(len(x)):
                fig.add_shape(
                    type="line",
                    x0=x[i], y0=y_pred[i],
                    x1=x[i], y1=y[i],
                    line=dict(color="red", width=1, dash="dot")
                )
        
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
    st.markdown(
        "The training error almost always decreases with complexity, but test error increases after the optimal model — this is overfitting."
    )

    x, y, y_true = get_poly_data(
        st.session_state.n_samples, st.session_state.noise, st.session_state.seed
    )
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        degree = st.slider("Polynomial Degree", 0, 15, 1)
        train_pct = st.slider("Training Set Size", 0.6, 0.9, 0.7, 0.05)
        
        # Re-split data only if controls change
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_pct, random_state=st.session_state.seed
        )
        
        # Sort for plotting
        sort_idx = np.argsort(x_train)
        x_train_s = x_train[sort_idx]
        y_train_s = y_train[sort_idx]

        sort_idx_test = np.argsort(x_test)
        x_test_s = x_test[sort_idx_test]
        y_test_s = y_test[sort_idx_test]

        metric_choice = st.selectbox(
            "Error Metric", 
            ("RMSE", "MSE", "MAE")
        )
        standardize = st.checkbox("Standardize x (recommended for high degrees)")

    # Loop to get errors for all degrees
    @st.cache_data
    def get_all_errors(train_pct, _x, _y, _seed, _standardize):
        # We pass _x, _y, _seed, _standardize to make cache aware of them
        x_train_loop, x_test_loop, y_train_loop, y_test_loop = train_test_split(
            _x, _y, train_size=train_pct, random_state=_seed
        )

        degrees = range(0, 16)
        train_errors = []
        test_errors = []

        for deg in degrees:
            steps = [('poly', PolynomialFeatures(degree=deg))]
            if _standardize:
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', LinearRegression()))
            
            pipeline = Pipeline(steps)
            pipeline.fit(x_train_loop.reshape(-1, 1), y_train_loop)
            
            y_train_pred = pipeline.predict(x_train_loop.reshape(-1, 1))
            y_test_pred = pipeline.predict(x_test_loop.reshape(-1, 1))
            
            if metric_choice == "RMSE":
                train_errors.append(np.sqrt(mean_squared_error(y_train_loop, y_train_pred)))
                test_errors.append(np.sqrt(mean_squared_error(y_test_loop, y_test_pred)))
            elif metric_choice == "MSE":
                train_errors.append(mean_squared_error(y_train_loop, y_train_pred))
                test_errors.append(mean_squared_error(y_test_loop, y_test_pred))
            else: # MAE
                train_errors.append(mean_absolute_error(y_train_loop, y_train_pred))
                test_errors.append(mean_absolute_error(y_test_loop, y_test_pred))
        
        return degrees, train_errors, test_errors

    degrees, train_errors, test_errors = get_all_errors(
        train_pct, x, y, st.session_state.seed, standardize
    )

    with col2:
        st.subheader("Fit Visualization (for selected degree)")
        
        # Fit model for *selected* degree
        steps_viz = [('poly', PolynomialFeatures(degree=degree))]
        if standardize:
            steps_viz.append(('scaler', StandardScaler()))
        steps_viz.append(('model', LinearRegression()))
        
        pipeline_viz = Pipeline(steps_viz)
        pipeline_viz.fit(x_train.reshape(-1, 1), y_train)

        # Create smooth line for prediction
        x_range = np.linspace(0, 1, 100).reshape(-1, 1)
        y_pred_range = pipeline_viz.predict(x_range)
        
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=x_train_s, y=y_train_s, mode='markers', name='Training Data', marker=dict(color='blue', opacity=0.7)))
        fig_fit.add_trace(go.Scatter(x=x_test_s, y=y_test_s, mode='markers', name='Test Data', marker=dict(color='red', opacity=0.7)))
        fig_fit.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='True Function (sin(2πx))', line=dict(color='green', dash='dash')))
        fig_fit.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred_range, mode='lines', name=f'Model (Degree {degree})', line=dict(color='orange', width=3)))
        fig_fit.update_layout(title="Model Fit vs. Data", xaxis_title="x", yaxis_title="y", yaxis_range=[-2.5, 2.5], height=400)
        st.plotly_chart(fig_fit, use_container_width=True)

        st.subheader(f"Error ({metric_choice}) vs. Polynomial Degree")
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=degrees, y=train_errors, mode='lines+markers', name='Train Error'))
        fig_err.add_trace(go.Scatter(x=degrees, y=test_errors, mode='lines+markers', name='Test Error'))
        
        # Highlight the minimum test error
        min_test_err_idx = np.argmin(test_errors)
        min_test_err_deg = degrees[min_test_err_idx]
        min_test_err_val = test_errors[min_test_err_idx]
        
        fig_err.add_annotation(
            x=min_test_err_deg, y=min_test_err_val,
            text=f"Min Test Error (Degree {min_test_err_deg})",
            showarrow=True, arrowhead=1, ax=20, ay=-40
        )
        fig_err.add_vline(x=min_test_err_deg, line=dict(color='red', dash='dot'))
        
        fig_err.update_layout(title="Train and Test Error vs. Model Complexity", xaxis_title="Polynomial Degree", yaxis_title=f"Error ({metric_choice})", height=400)
        st.plotly_chart(fig_err, use_container_width=True)

# ==============================================================================
# --- Tab 3: Logistic Regression ---
# ==============================================================================
with tab3:
    st.header("Tab 3: Logistic Regression")
    st.markdown(
        "Adjust the decision threshold to see how the trade-off between precision and recall changes."
    )

    X, y = get_logistic_data(
        st.session_state.n_samples, st.session_state.noise, st.session_state.seed
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=st.session_state.seed, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        C = st.select_slider(
            "Regularization Parameter (C)",
            options=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            value=1.0,
            help="Smaller C = stronger regularization."
        )
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        
        # Fit model
        model = LogisticRegression(C=C, random_state=st.session_state.seed, solver='liblinear')
        model.fit(X_train_scaled, y_train)
        
        # Get probabilities and predictions
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Metrics
        st.subheader("Metrics (at current threshold)")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        m_col2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
        m_col3, m_col4 = st.columns(2)
        m_col3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
        m_col4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
        st.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

        st.subheader("Model Coefficients")
        st.write(f"Feature 1 Coef: {model.coef_[0][0]:.3f}")
        st.write(f"Feature 2 Coef: {model.coef_[0][1]:.3f}")
        st.write(f"Intercept: {model.intercept_[0]:.3f}")
        
    with col2:
        st.subheader("Decision Boundary and Test Data")
        
        # Create meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        # Predict on grid (must scale grid points)
        grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        Z = model.predict_proba(grid_scaled)[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot data points
        df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2'])
        df_test['Class'] = y_test
        df_test['Class'] = df_test['Class'].astype(str) # For discrete colors
        
        fig_bound = px.scatter(df_test, x='Feature 1', y='Feature 2', color='Class',
                               color_discrete_map={'0': 'blue', '1': 'red'})
        
        # Add decision boundary contour
        fig_bound.add_trace(
            go.Contour(
                x=np.arange(x_min, x_max, 0.02),
                y=np.arange(y_min, y_max, 0.02),
                z=Z,
                name="Prob(y=1)",
                showscale=False,
                contours=dict(
                    start=0, end=1, size=0.1,
                ),
                opacity=0.5,
                line_width=0,
            )
        )
        
        # Add contour for the *current threshold*
        fig_bound.add_trace(
            go.Contour(
                x=np.arange(x_min, x_max, 0.02),
                y=np.arange(y_min, y_max, 0.02),
                z=Z,
                showscale=False,
                contours_coloring='lines',
                contours=dict(
                    start=threshold, end=threshold, size=0
                ),
                line=dict(color='black', width=3, dash='dash'),
                name=f"Threshold ({threshold})"
            )
        )
        
        fig_bound.update_layout(title="Decision Boundary", height=500)
        st.plotly_chart(fig_bound, use_container_width=True)

    # --- Row for ROC, PR, and Confusion Matrix ---
    st.subheader("Model Performance Curves")
    plot_col1, plot_col2, plot_col3 = st.columns(3)
    
    with plot_col1:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc_score(y_test, y_proba):.3f})")
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='Gray', dash='dash'))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with plot_col2:
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig_pr = px.line(x=recall, y=precision, title="Precision-Recall Curve")
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=400)
        st.plotly_chart(fig_pr, use_container_width=True)
        
    with plot_col3:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"), 
                           x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'],
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)