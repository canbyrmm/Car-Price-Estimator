import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# CSV'den veri oku (eğitim sırasında kullanılan veriyle aynı olmalı)
df = pd.read_csv("data.csv")

# Sayısal olmayan power_ps değerlerini temizle
def extract_numeric_power(x):
    try:
        return int(float(str(x).replace(',', '.')))
    except:
        return None

df["clean_power"] = df["power_ps"].apply(extract_numeric_power)

st.set_page_config(layout="centered")
st.image("can_white.png", use_column_width=False, width=250)

st.title("🚗 Car Price Estimator - Can Auto")

st.markdown("""
This app provides two pricing modes:

- **1. Market Estimate**: Predicts your car's approximate market value.
- **2. Can Auto Offer**: Simulates a dealer offer (below market price).
""")

option = st.radio("Select Prediction Type:", ("Market Value", "Can Auto Offer"))

# Marka secimi
all_brands = sorted(df["brand"].dropna().unique())
selected_brand = st.selectbox("Brand", all_brands)

# Model secimi (markaya bagli)
filtered_models = df[df["brand"] == selected_brand]["model"].dropna().unique()
selected_model = st.selectbox("Model", sorted(filtered_models))

# Fuel Type secimi
fuel_options = df[(df["brand"] == selected_brand) & (df["model"] == selected_model)]["fuel_type"].dropna().unique()
valid_fuels = [f for f in fuel_options if f.lower() in ["diesel", "petrol", "electric", "hybrid", "other", "unknown"]]
if not valid_fuels:
    valid_fuels = ["Petrol", "Diesel", "Electric", "Hybrid", "Other"]
fuel = st.selectbox("Fuel Type", sorted(valid_fuels))

# Transmission secimi
gear_options = df[(df["brand"] == selected_brand) & (df["model"] == selected_model)]["transmission_type"].dropna().unique()
valid_gears = [g for g in gear_options if g.lower() in ["manual", "automatic", "semi-automatic"]]
if not valid_gears:
    valid_gears = ["Manual", "Automatic", "Semi-automatic"]
gear = st.selectbox("Transmission Type", sorted(valid_gears))

# Power (PS) secimi - modele göre aralıktan seçmeli
power_values = df[(df["brand"] == selected_brand) & (df["model"] == selected_model)]["clean_power"].dropna().unique()
power_values = sorted(list(set(power_values)))

if power_values:
    power = st.selectbox("Power (PS)", power_values)
else:
    power = st.number_input("Power (PS)", min_value=1, max_value=1000, value=120)

# Year ve KM bilgisi
year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2018)
km = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=100000)

# Color
colors = df["color"].dropna().unique()
color = st.selectbox("Color", sorted(colors))

# Feature engineering
car_age = 2025 - year
km_per_year = km / (car_age + 1)
is_low_mileage = 1 if km < 50000 else 0
is_new_car = 1 if car_age <= 2 else 0

# 🎯 Modeli yukle
model = joblib.load("final_rf_model.joblib")

if st.button("Estimate Price"):
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

        # Grafik yerine görsel gösterimi
        if os.path.exists("cycle.png"):
            st.image("cycle.png", use_column_width=True)
        # Altına fiyatları yazdır
        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
            🟢 <b>Easy Sale:</b> €{predicted_price*0.90:,.0f} &nbsp;&nbsp;&nbsp; 
            🟡 <b>Fair Market:</b> €{predicted_price:,.0f} &nbsp;&nbsp;&nbsp; 
            🔴 <b>Hard to Sell:</b> €{predicted_price*1.10:,.0f}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.image("CAR.png", width=250)
        st.markdown(f"### 🚘 CAN Auto offers you: **€{predicted_price*0.85:,.2f}**")
