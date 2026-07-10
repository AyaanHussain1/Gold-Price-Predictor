# Gold Price Prediction System

🔗 **Live Demo:** [Add your deployment link here]

A machine learning web app that predicts the target Gold Price (GLD) based on key global market indicators, trained on historical financial data.

## Overview
Gold prices are heavily influenced by broader macroeconomic factors like stock market performance, oil prices, silver valuation, and currency strength. This project processes a comprehensive historical dataset containing continuous financial metrics, performs exploratory data analysis (EDA), and trains a tree-based machine learning model to estimate the price of Gold (GLD). The deployment features an interactive Streamlit user interface with dynamically bounded sliders constructed directly from dataset metrics to ensure intuitive real-time predictions.

## How It Works
1. **Data** — Utilizes a historical financial market dataset containing 2,290 records with 5 core continuous features: SPX (S&P 500 Index), GLD (Gold Trust), USO (United States Oil Fund), SLV (Silver Trust), and the EUR/USD Exchange Rate.
2. **Data Integrity** — Validated dataset completeness through missing-value checks (`df.isnull().sum()`) and duplicate audits (`df.duplicated().sum()`), confirming a clean, high-fidelity structure.
3. **Feature Splitting** — Separated features by isolating the `Date` column and the target variable `GLD`. Split the remaining indicator matrices into an 80/20 train-test ratio using `train_test_split` to guarantee reliable evaluation.
4. **Model Architecture** — Trained a non-linear `DecisionTreeRegressor` configured with a maximum depth of 100 to robustly map the multi-dimensional indicators to the gold pricing target.
5. **Interactive Interface** — Implemented an optimization pipeline via Streamlit's `@st.cache_resource` to keep data loading and model initialization cached in memory for instant query execution.
6. **Deployment** — Designed a dual-column dashboard UI using Streamlit sliders. The minimum, maximum, and default slider positions adaptively sync with the mathematical limits and means extracted directly from the asset dataset.

## Tech Stack
- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling & Validation:** Scikit-learn (`DecisionTreeRegressor`, `train_test_split`)
- **Visualization (EDA):** Matplotlib, Seaborn
- **App/Deployment:** Streamlit

## Project Structure
```text
gold-price-prediction/
├── main.ipynb            # Data inspection, exploratory analysis, and experimental training
├── main.py               # Streamlit application containing data caching, UI sliders, and model routing
├── gold_price_data.csv   # Historical market data (SPX, GLD, USO, SLV, EUR/USD)
└── requirements.txt      # Python library dependencies
