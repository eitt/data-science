Here is your updated `README.md` file.

I have completely overhauled it to match the final 9-page structure of your application, updated the dependencies, and added the author profile as you requested.

-----

# Understanding Supervised Learning: An Interactive Streamlit App

This is an interactive Streamlit web application designed to teach the fundamental concepts of supervised machine learning. It provides hands-on, visual demonstrations of optimization, regression, classification, and neural networks.

This app is built for students, data science enthusiasts, and anyone looking for an intuitive way to understand how machine learning models actually *learn*.

## About the Author

This application was designed and developed by **Dr. Leonardo H. Talero-Sarmiento**.

Dr. Talero is a Ph.D. in Engineering from the Universidad Autónoma de Bucaramanga, with expertise in mathematical modeling, data analytics, operations research, manufacturing systems, process improvement and technology adoption. His research addresses decision-making and production-planning challenges in agricultural and industrial contexts, applying operations research, bibliometric analysis, systematic reviews, and machine-learning methods to strengthen value-chain resilience, optimize healthcare delivery, and drive digital transformation.

He regularly publishes in leading peer-reviewed journals, including *Heliyon, Cogent Engineering, Ecological Informatics, AgriEngineering, Big Data and Cognitive Computing, Digital Policy, Regulation and Governance, Revista Colombiana de Computación, Suma de Negocios, IngeCUC, Apuntes del Cenes, Estudios Gerenciales* and *Contaduría y Administración*.

## Learning Modules

The app guides you through a logical progression of topics, from core theory to advanced models.

### 1\. Introduction

  * **Goal:** Welcome page and introduction to the app's author and purpose.

### 2\. Gradient Descent

  * **Goal:** Understand the core optimization algorithm that powers most of machine learning.
  * **Interactivity:**
      * Visualize a 3D "cost surface" (a sandpit).
      * Set a starting point and a **learning rate**.
      * Watch the algorithm take steps downhill to find the "lowest point."
      * See how a high learning rate causes the algorithm to "overshoot" and fail.

### 3\. Manual Linear Fit

  * **Goal:** Connect the abstract concept of Gradient Descent to a real model.
  * **Interactivity:**
      * **Model Space:** Manually adjust the **slope (β₁)** and **intercept (β₀)** of a line to fit data points.
      * **Parameter Space:** Simultaneously, watch your `(β₁, β₀)` position move on a 2D "cost surface" plot.
      * **Key Takeaway:** See that "fitting the line" is the same as "finding the bottom of the valley" in the cost surface.

### 4\. Train vs Test (Overfitting)

  * **Goal:** Learn why model complexity is a trade-off and what overfitting looks like.
  * **Interactivity:**
      * Fit a **polynomial model** to a synthetic dataset (e.g., a sine wave).
      * Adjust the **polynomial degree** (model complexity) with a slider.
      * Analyze a separate plot that shows the **Training Error** vs. the **Test Error**.
      * **Key Takeaway:** Watch as the training error continuously drops, but the test error starts to rise after a certain point—this is **overfitting**.

### 5\. Logistic Regression

  * **Goal:** Introduce binary classification, log-odds, and odds ratios.
  * **Interactivity:**
      * Visualize the **decision boundary** on a 2D dataset.
      * Adjust the **decision threshold** (0 to 1) and see its effect on the **confusion matrix** and metrics like Precision and Recall.
      * Explore a real-world **manufacturing failure** example to understand how to interpret model coefficients.

### 6\. Decision Trees

  * **Goal:** Learn an alternative classification method that uses rules and impurity.
  * **Interactivity:**
      * Use the same **manufacturing failure** dataset.
      * Understand **Gini Impurity** and **Entropy**.
      * Change the **Tree Max Depth** and impurity criterion.
      * Watch the decision tree visualization update in real-time.

### 7\. Neural Network Training

  * **Goal:** Understand how Gradient Descent (Backpropagation) trains a complex, non-linear model.
  * **Interactivity:**
      * Design your own neural network by setting the **number of hidden layers** and **neurons per layer**.
      * See the **total trainable parameters** (weights & biases) calculated instantly.
      * Click "Train Model" and watch two plots animate:
        1.  The **model's prediction** fitting the data.
        2.  The **loss curve** going down, showing the model "learning."

### 8\. Time Series (Data Leaks)

  * **Goal:** Learn about a critical and common pitfall in time series modeling.
  * **Interactivity:**
      * Compare two models:
        1.  **Incorrect Model:** Uses a *random* train/test split (a data leak).
        2.  **Correct Model:** Uses a *time-based* train/test split.
      * See how the data leak produces a deceptively low (and wrong) error metric.

### 9\. TS Analysis

  * **Goal:** Learn a simple, fast technique for analyzing time series data.
  * **Interactivity:**
      * Load sample datasets (e.g., "Air Passengers").
      * Adjust a **Rolling Window Size** slider.
      * See the **Rolling Average** (trend) and **Rolling Standard Deviation** (volatility) update on the chart.

## How to Run

Follow these steps to run the app on your local machine.

1.  **Prerequisites:**

      * Python 3.9+ (Python 3.11 recommended)
      * `pip` (Python package installer)

2.  **Clone the Repository (or Download Files):**

      * If this is a git repository:
        ```bash
        git clone <your-repo-url>
        cd <your-repo-directory>
        ```
      * Alternatively, just download all the `.py`, `.txt`, and `.png` files into a new folder.

3.  **Create a Virtual Environment (Recommended):**

    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    venv\Scripts\activate
    ```

4.  **Install Dependencies:**

      * With the virtual environment activated, install the required libraries:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Run the Streamlit App:**

      * Run the following command in your terminal:
        ```bash
        streamlit run app.py
        ```

6.  **View the App:**

      * Streamlit will automatically open a new tab in your default web browser. If it doesn't, the terminal will provide a local URL (usually `http://localhost:8501`) that you can open manually.

## Project Structure

```
├── app.py                  # The main Streamlit application code
├── requirements.txt        # The list of Python dependencies
├── runtime.txt             # Specifies the Python version for Streamlit Cloud

```

### `requirements.txt`

```
numpy
pandas
plotly
scikit-learn
streamlit
matplotlib
```