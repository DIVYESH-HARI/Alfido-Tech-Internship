import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 🔁 Exchange Rate (Update as needed — 1 USD = ₹83.5 approx in 2025)
USD_TO_INR = 83.5

# Load model and features
try:
    model = joblib.load('house_price_model.pkl')
    features = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("❌ Model files not found! Please run the Jupyter notebook first to generate `house_price_model.pkl` and `model_features.pkl`.")
    st.stop()

st.title("🏠 House Price Prediction")
st.write("Enter house features to predict price in **USD** and **INR**.")

# Input form
bedrooms = st.slider("🛏️ Bedrooms", 1, 10, 3)
bathrooms = st.slider("🚽 Bathrooms", 0.5, 8.0, 2.0, step=0.25)
sqft_living = st.number_input("🏠 Living Area (sqft)", 500, 10000, 1500)
sqft_lot = st.number_input("🌳 Lot Size (sqft)", 500, 1000000, 8000)
floors = st.slider(" Floors", 1.0, 3.5, 1.5, step=0.5)
waterfront = st.checkbox("🌊 Waterfront?")
view = st.slider("👁️ View (0–4)", 0, 4, 0)
condition = st.slider("🔧 Condition (1–5)", 1, 5, 3)
sqft_above = st.number_input("⬆️ Above-ground sqft", 500, 10000, sqft_living)
sqft_basement = st.number_input("⬇️ Basement sqft", 0, 5000, max(0, sqft_living - sqft_above))
yr_built = st.number_input("📅 Year Built", 1900, 2025, 1980)
yr_renovated = st.number_input("🔧 Year Renovated (0 if none)", 0, 2025, 0)

# Derived features
age = 2025 - yr_built
is_renovated = 1 if yr_renovated != 0 and yr_renovated != yr_built else 0
if yr_renovated == 0:
    yr_renovated = yr_built  # match training logic

# Create input DataFrame
input_data = pd.DataFrame([[
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    int(waterfront), view, condition, sqft_above, sqft_basement,
    yr_built, yr_renovated, age, is_renovated
]], columns=features)

# Predict
if st.button("🔮 Predict Price"):
    prediction_usd = model.predict(input_data)[0]
    prediction_inr = prediction_usd * USD_TO_INR

    st.success(f"**Predicted House Price**")
    st.metric(label="💵 USD", value=f"${prediction_usd:,.2f}")
    st.metric(label="🇮🇳 INR", value=f"₹{prediction_inr:,.2f}")
    
    # Optional: show exchange rate used
    st.caption(f"💱 Using exchange rate: $1 = ₹{USD_TO_INR}")
    st.info("💡 Note: This is a simple linear model — actual prices may vary. Exchange rate is approximate.")