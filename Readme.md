# Gold Price Prediction System

🔗 Live Demo: [https://gold-price-predictor-md.streamlit.app/]

A machine learning web app that predicts the target Gold Price (GLD) based on key global market indicators, trained on historical financial data.

## Overview
Gold prices are heavily influenced by broader macroeconomic factors like stock market performance, oil prices, silver valuation, and currency strength. This project processes a comprehensive historical dataset containing continuous financial metrics, performs exploratory data analysis (EDA), and trains a tree-based machine learning model to estimate the price of Gold (GLD). The finalized standalone model pipeline is serialized into a reusable tracking file (`model.pkl`), allowing the Streamlit web application to initialize instantly and perform real-time evaluations without re-running data splits during runtime.

## How It Works
1. **Data** — Utilizes a historical financial market dataset containing 2,290 records with 5 core continuous features: SPX (S&P 500 Index), GLD (Gold Trust), USO (United States Oil Fund), SLV (Silver Trust), and the EUR/USD Exchange Rate.
2. **Data Integrity** — Validated dataset completeness through missing-value checks (`df.isnull().sum()`) and duplicate audits (`df.duplicated().sum()`), confirming a clean, high-fidelity structure.
3. **Feature Splitting** — Separated features by isolating the `Date` column and the target variable `GLD`. Split the remaining indicator matrices into an 80/20 train-test ratio using `train_test_split` to guarantee reliable evaluation.
4. **Model Architecture** — Trained a non-linear `DecisionTreeRegressor` configured with a maximum depth of 100 to robustly map the multi-dimensional indicators to the gold pricing target.
5. **Serialization** — Exported the high-accuracy trained model object directly into a serialized binary format (`model.pkl`) to isolate training computation from downstream user traffic pipelines.
6. **Deployment & Interface** — Built an interactive Streamlit UI using memory caching via `@st.cache_resource`. The web app instantly processes manual user inputs via dual-column slider frameworks dynamically bounded to the dataset's original statistical boundaries.

## Tech Stack
- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling & Serialization:** Scikit-learn (`DecisionTreeRegressor`), Pickle
- **Visualization (EDA):** Matplotlib, Seaborn
- **App/Deployment:** Streamlit

## Project Structure
```text
gold-price-prediction/
├── main.ipynb            # Data inspection, exploratory analysis, and model validation
├── main.py               # Streamlit web app containing caching, UI sliders, and inference logic
├── model.pkl             # Serialized, trained Decision Tree Regressor model file
├── gold_price_data.csv   # Historical market dataset (SPX, GLD, USO, SLV, EUR/USD)
└── requirements.txt      # Python library dependencies
