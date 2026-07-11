"""
app.py
Aplikasi Gradio: Prediksi Harga Mobil Bekas
"""

import json
import joblib
import pandas as pd
import gradio as gr

# Kurs konversi USD -> IDR (dataset asli berharga dalam USD).
# Ubah angka ini kapan saja sesuai kurs terbaru.
KURS_USD_TO_IDR = 16000


def format_rupiah(angka: float) -> str:
    """Format angka jadi 'Rp 123.456.789' gaya Indonesia."""
    return "Rp " + f"{angka:,.0f}".replace(",", ".")


# ---- Load model & opsi fitur (sekali saat aplikasi start) ----
pipeline = joblib.load("model/model.pkl")

with open("model/feature_options.json", "r") as f:
    options = json.load(f)

brand_model_map = options["brand_model_map"]
best_model_name = options.get("best_model_name", "Regresi")

year_min = int(options["numeric_ranges"]["Year"]["min"])
year_max = 2026
engine_min = float(options["numeric_ranges"]["Engine Size"]["min"])
engine_max = float(options["numeric_ranges"]["Engine Size"]["max"])
mileage_max = float(options["numeric_ranges"]["Mileage"]["max"]) * 1.2


def update_model_choices(brand):
    """Dipanggil setiap kali dropdown Brand berubah -> update pilihan Model."""
    model_choices = brand_model_map.get(brand, options["Model"])
    return gr.update(choices=model_choices, value=model_choices[0])


def predict_price(brand, model_name, fuel_type, transmission, condition, year, engine_size, mileage):
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

    hasil = f"## Estimasi Harga: {format_rupiah(prediction_idr)}\n\n(kurs yang dipakai: 1 USD = {format_rupiah(KURS_USD_TO_IDR)})"
    return hasil, input_df


with gr.Blocks(title="Prediksi Harga Mobil Bekas") as demo:
    gr.Markdown("# 🚗 Prediksi Harga Mobil Bekas")
    gr.Markdown(
        f"Aplikasi ini mengestimasi harga jual mobil bekas berdasarkan spesifikasinya, "
        f"menggunakan model **{best_model_name}**."
    )
    gr.Markdown("---")
    gr.Markdown("### Masukkan Spesifikasi Mobil")

    with gr.Row():
        with gr.Column():
            brand_input = gr.Dropdown(choices=options["Brand"], label="Brand", value=options["Brand"][0])
            model_input = gr.Dropdown(
                choices=brand_model_map.get(options["Brand"][0], options["Model"]),
                label="Model",
                value=brand_model_map.get(options["Brand"][0], options["Model"])[0],
            )
            fuel_input = gr.Dropdown(choices=options["Fuel Type"], label="Fuel Type", value=options["Fuel Type"][0])
            transmission_input = gr.Dropdown(choices=options["Transmission"], label="Transmission", value=options["Transmission"][0])

        with gr.Column():
            condition_input = gr.Dropdown(choices=options["Condition"], label="Condition", value=options["Condition"][0])
            year_input = gr.Number(label="Year", minimum=year_min, maximum=year_max, value=2018, precision=0)
            engine_input = gr.Number(label="Engine Size (L)", minimum=engine_min, maximum=engine_max, value=2.5, step=0.1)
            mileage_input = gr.Number(label="Mileage (km)", minimum=0.0, maximum=mileage_max, value=45000.0, step=1000.0)

    # Dropdown Model otomatis menyesuaikan saat Brand diganti
    brand_input.change(fn=update_model_choices, inputs=brand_input, outputs=model_input)

    predict_btn = gr.Button("🔮 Prediksi Harga", variant="primary")

    output_text = gr.Markdown()
    with gr.Accordion("Lihat detail input", open=False):
        output_table = gr.Dataframe()

    predict_btn.click(
        fn=predict_price,
        inputs=[brand_input, model_input, fuel_input, transmission_input, condition_input, year_input, engine_input, mileage_input],
        outputs=[output_text, output_table],
    )

    gr.Markdown("---")
    gr.Markdown(
        "Model dilatih menggunakan dataset Car Price Prediction (Kaggle) yang berharga dalam USD, "
        "lalu dikonversi ke Rupiah. Prediksi bersifat estimasi dan tidak menggantikan penilaian profesional."
    )

if __name__ == "__main__":
    demo.launch()
