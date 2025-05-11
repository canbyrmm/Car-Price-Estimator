import os
import gdown
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Download data and model files from Google Drive if not already present
if not os.path.exists("data.csv"):
    gdown.download("https://drive.google.com/uc?id=1WuewLhVJs6hZLWsm53H6j4AGjoIaVtMK", "data.csv", quiet=False)

if not os.path.exists("final_rf_model.joblib"):
    gdown.download("https://drive.google.com/uc?id=1M_KpNCEMfRc40l6JCfGBNO4KB34Qfl4C", "final_rf_model.joblib", quiet=False)

# Load CSV dataset
df = pd.read_csv("data.csv")

# Clean the power_ps column and convert valid numeric strings to integers
def extract_numeric_power(x):
    try:
        return int(float(str(x).replace(',', '.')))
    except:
        return None

df["clean_power"] = df["power_ps"].apply(extract_numeric_power)

col1, col2 = st.columns([1, 3])

with col1:
    st.image("can_white.png", width=200)  # daha küçük tut

with col2:
    st.markdown("<h1 style='text-align: center;'>Car Price Estimator - Can Auto</h1>", unsafe_allow_html=True)


st.markdown("""
This app provides two pricing modes:

- **1. Market Estimate**: Predicts your car's approximate market value.
- **2. Can Auto Offer**: Simulates a dealer offer (below market price).
""")

option = st.radio("Select Prediction Type:", ("Market Value", "Can Auto Offer"))

# Brand selection
all_brands = sorted(df["brand"].dropna().unique())
selected_brand = st.selectbox("Brand", all_brands)

# Model selection based on selected brand
filtered_models = df[df["brand"] == selected_brand]["model"].dropna().unique()
selected_model = st.selectbox("Model", sorted(filtered_models))

# Filter available fuel types for selected brand and model
fuel_options = df[(df["brand"] == selected_brand) & (df["model"] == selected_model)]["fuel_type"].dropna().unique()
valid_fuels = [f for f in fuel_options if f.lower() in ["diesel", "petrol", "electric", "hybrid", "other", "unknown"]]
if not valid_fuels:
    valid_fuels = ["Petrol", "Diesel", "Electric", "Hybrid", "Other"]
fuel = st.selectbox("Fuel Type", sorted(valid_fuels))

# Filter available transmission types for selected brand and model
gear_options = df[(df["brand"] == selected_brand) & (df["model"] == selected_model)]["transmission_type"].dropna().unique()
valid_gears = [g for g in gear_options if g.lower() in ["manual", "automatic", "semi-automatic"]]
if not valid_gears:
    valid_gears = ["Manual", "Automatic", "Semi-automatic"]
gear = st.selectbox("Transmission Type", sorted(valid_gears))

# Power (PS) selection based on brand and model
power_values = df[(df["brand"] == selected_brand) & (df["model"] == selected_model)]["clean_power"].dropna().unique()
power_values = sorted(list(set(power_values)))

if power_values:
    power = st.selectbox("Power (PS)", power_values)
else:
    power = st.number_input("Power (PS)", min_value=1, max_value=1000, value=120)

# Year and mileage inputs
year = st.number_input("Year of Manufacture", min_value=1980, max_value=2023, value=2018)
km = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=100000)

# Color selection
colors = df["color"].dropna().unique()
color = st.selectbox("Color", sorted(colors))

# Load the trained model
model = joblib.load("final_rf_model.joblib")

if st.button("Estimate Price"):
    car_age = 2025 - year
    km_per_year = km / (car_age + 1)
    is_low_mileage = 1 if km < 50000 else 0
    is_new_car = 1 if car_age <= 2 else 0

    input_df = pd.DataFrame([{
        "brand": selected_brand,
        "model": selected_model,
        "fuel_type": fuel,
        "transmission_type": gear,
        "color": color,
        "year": year,
        "power_ps": power,
        "mileage_in_km": km,
        "car_age": car_age,
        "km_per_year": km_per_year,
        "is_low_mileage": is_low_mileage,
        "is_new_car": is_new_car
    }])

    predicted_price = model.predict(input_df)[0]

    if option == "Market Value":
        # Display estimated price and visual chart
        if os.path.exists("cycle.png"):
            st.image("cycle.png", use_container_width=True)
        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
            🟢 <b>Easy Sale:</b> €{predicted_price*0.90:,.0f} &nbsp;&nbsp;&nbsp; 
            🟡 <b>Fair Market:</b> €{predicted_price:,.0f} &nbsp;&nbsp;&nbsp; 
            🔴 <b>Hard to Sell:</b> €{predicted_price*1.10:,.0f}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.image("CAR.png", width=200)
        st.markdown(f"### 🚘 CAN Auto offers you: **€{predicted_price*0.85:,.2f}**")
