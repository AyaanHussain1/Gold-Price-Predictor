import streamlit as st
import pandas as pd
import numpy as np
import os  # Added to handle absolute paths during deployment
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle 
# 1. Page Configuration
st.set_page_config(page_title="Gold Price Predictor", layout="centered")
st.title(" Gold Price Prediction System")
st.markdown("Adjust the market indicators below to predict the target Gold Price (GLD).")

# 2. Load Data and Train Model (Cached so it runs instantly)
@st.cache_resource
def load_data_and_train():
    # Dynamic path handling: Points directly to the directory where this script runs from
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "gold_price_data.csv")
    
    # Read the data securely using the constructed path
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=["Date", "GLD"])
    y = df["GLD"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=100, random_state=2)
    model.fit(X_train, y_train)
    with open ("model.pkl","wb") as f:
        pickle.dump(model,f)
    # Get min and max limits for sliders dynamically from your data
    limits = {col: (float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in X.columns}
    
    return model, limits

# Initialize model and slider ranges
model, limits = load_data_and_train()

# Create the UI Boxes/Sliders dynamically
st.header("Market Indicators")

# Creating layout columns for a cleaner UI look
col1, col2 = st.columns(2)

with col1:
    spx_min, spx_max, spx_mean = limits["SPX"]
    spx = st.slider("SPX (S&P 500 Index)", min_value=spx_min, max_value=spx_max, value=spx_mean, step=0.1)

    uso_min, uso_max, uso_mean = limits["USO"]
    uso = st.slider("USO (United States Oil Fund)", min_value=uso_min, max_value=uso_max, value=uso_mean, step=0.1)

with col2:
    slv_min, slv_max, slv_mean = limits["SLV"]
    slv = st.slider("SLV (Silver Trust)", min_value=slv_min, max_value=slv_max, value=slv_mean, step=0.1)

    eur_usd_min, eur_usd_max, eur_usd_mean = limits["EUR/USD"]
    eur_usd = st.slider("EUR/USD Exchange Rate", min_value=eur_usd_min, max_value=eur_usd_max, value=eur_usd_mean, step=0.001)

st.markdown("---")

# Prediction Trigger Button
if st.button("Predict Gold Price", type="primary"):
    # Format input exactly like your test sample, passing the explicit column names to avoid the UserWarning!
    input_df = pd.DataFrame([[spx, uso, slv, eur_usd]], columns=["SPX", "USO", "SLV", "EUR/USD"])
    
    # Predict
    prediction = model.predict(input_df)
    
    # Display Result nicely
    st.success(f"### Predicted Gold Price (GLD): **${prediction[0]:.2f}**")