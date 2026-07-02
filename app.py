"""
app.py
Aplikasi Streamlit: Prediksi Harga Mobil Bekas
"""

import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗", layout="centered")

# Kurs konversi USD -> IDR (dataset asli berharga dalam USD).
# Ubah angka ini kapan saja sesuai kurs terbaru.
KURS_USD_TO_IDR = 16000


def format_rupiah(angka: float) -> str:
    """Format angka jadi 'Rp 123.456.789' gaya Indonesia."""
    return "Rp " + f"{angka:,.0f}".replace(",", ".")


@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")


@st.cache_data
def load_feature_options():
    with open("model/feature_options.json", "r") as f:
        return json.load(f)


pipeline = load_model()
options = load_feature_options()
brand_model_map = options["brand_model_map"]

st.title("🚗 Prediksi Harga Mobil Bekas")
st.write(
    "Aplikasi ini mengestimasi harga jual mobil bekas berdasarkan spesifikasinya, "
    f"menggunakan model **{options.get('best_model_name', 'Regresi')}**."
)

st.divider()
st.subheader("Masukkan Spesifikasi Mobil")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options["Brand"])

    # Dropdown Model mengikuti Brand yang dipilih
    model_choices = brand_model_map.get(brand, options["Model"])
    model_name = st.selectbox("Model", model_choices)

    fuel_type = st.selectbox("Fuel Type", options["Fuel Type"])
    transmission = st.selectbox("Transmission", options["Transmission"])

with col2:
    condition = st.selectbox("Condition", options["Condition"])
    year = st.number_input(
        "Year",
        min_value=int(options["numeric_ranges"]["Year"]["min"]),
        max_value=2026,
        value=2018,
        step=1,
    )
    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=float(options["numeric_ranges"]["Engine Size"]["min"]),
        max_value=float(options["numeric_ranges"]["Engine Size"]["max"]),
        value=2.5,
        step=0.1,
    )
    mileage = st.number_input(
        "Mileage (km)",
        min_value=0.0,
        max_value=float(options["numeric_ranges"]["Mileage"]["max"]) * 1.2,
        value=45000.0,
        step=1000.0,
    )

st.divider()

if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Year": year,
        "Engine Size": engine_size,
        "Mileage": mileage,
        "Brand": brand,
        "Fuel Type": fuel_type,
        "Transmission": transmission,
        "Condition": condition,
        "Model": model_name,
    }])

    prediction_usd = pipeline.predict(input_df)[0]
    prediction_idr = prediction_usd * KURS_USD_TO_IDR

    st.success(f"### Estimasi Harga: {format_rupiah(prediction_idr)}")
    st.caption(f"(kurs yang dipakai: 1 USD = {format_rupiah(KURS_USD_TO_IDR)})")

    with st.expander("Lihat detail input"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Model dilatih menggunakan dataset Car Price Prediction (Kaggle) yang berharga dalam USD, "
    "lalu dikonversi ke Rupiah. Prediksi bersifat estimasi dan tidak menggantikan penilaian profesional."
)
