import json
import joblib
import pandas as pd

MODEL_PATH = "model/model.pkl"
OPTIONS_PATH = "model/feature_options.json"
KURS_USD_TO_IDR = 16000  # ubah sesuai kurs terbaru bila perlu


def load_model(model_path: str = MODEL_PATH):
    """Memuat pipeline model yang sudah dilatih."""
    return joblib.load(model_path)


def load_feature_options(options_path: str = OPTIONS_PATH) -> dict:
    """Memuat metadata fitur (opsi kategori & rentang numerik) untuk UI."""
    with open(options_path, "r") as f:
        return json.load(f)


def predict_price(
    brand: str,
    year: int,
    engine_size: float,
    fuel_type: str,
    transmission: str,
    mileage: float,
    condition: str,
    model_name: str,
    pipeline=None,
) -> float:
    """
    Memprediksi harga mobil bekas berdasarkan spesifikasi input.

    Mengembalikan estimasi harga (float).
    """
    if pipeline is None:
        pipeline = load_model()

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
    return float(prediction_usd) * KURS_USD_TO_IDR


if __name__ == "__main__":
    # Contoh penggunaan skrip inferensi secara langsung
    contoh_harga = predict_price(
        brand="Toyota",
        year=2018,
        engine_size=2.5,
        fuel_type="Petrol",
        transmission="Automatic",
        mileage=45000,
        condition="Used",
        model_name="Corolla",
    )
    print(f"Estimasi harga mobil: Rp {contoh_harga:,.0f}".replace(",", "."))
