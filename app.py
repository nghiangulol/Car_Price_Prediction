import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


model = joblib.load("car_price_predictor.pkl")
df = pd.read_csv("CarsData_Cleaned.csv")


st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="",
    layout="centered"
)

st.title("Used Car Price Prediction")
st.write("Enter the vehicle information below to estimate its selling price.")



manufacturers = sorted(df["Manufacturer"].dropna().unique())

manufacturer = st.selectbox(
    "Manufacturer",
    manufacturers
)

available_models = sorted(
    df.loc[
        df["Manufacturer"] == manufacturer,
        "model"
    ].dropna().unique()
)

car_model = st.selectbox(
    "Model",
    available_models
)


transmissions = sorted(
    df["transmission"].dropna().unique()
)

transmission = st.selectbox(
    "Transmission",
    transmissions
)


fuel_types = sorted(
    df["fuelType"].dropna().unique()
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_types
)

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=datetime.now().year,
    value=2020
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=50000,
    step=1000
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=50.0
)

engine_size = st.number_input(
    "Engine Size",
    min_value=0.0,
    value=1.5,
    step=0.1
)

tax = st.number_input(
    "Tax",
    min_value=0,
    value=150
)

if st.button("Predict Price"):

    usage_years = datetime.now().year - year
    mileage_log = np.log1p(mileage)

    input_df = pd.DataFrame({
        "model": [car_model],
        "transmission": [transmission],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size],
        "Manufacturer": [manufacturer],
        "usage_years": [usage_years],
        "mileage_log": [mileage_log]
    })

    pred_log = model.predict(input_df)
    
    predicted_price = np.expm1(pred_log[0])

    st.success(
        f"Estimated Price: £{predicted_price:,.0f}"
    )

    st.write("### Vehicle Summary")

    st.write(f"**Manufacturer:** {manufacturer}")
    st.write(f"**Model:** {car_model}")
    st.write(f"**Year:** {year}")
    st.write(f"**Mileage:** {mileage:,}")