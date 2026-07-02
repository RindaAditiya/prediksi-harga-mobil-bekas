# Prediksi Harga Mobil Bekas (Used Car Price Prediction)

Proyek tugas akhir mata kuliah Fundamen Sains Data — Aplikasi Supervised Learning.

## 1. Deskripsi Proyek

- **Kasus**: Regresi — memprediksi harga jual mobil bekas berdasarkan spesifikasinya.
- **Dataset**: [Car Price Prediction (Kaggle)](https://www.kaggle.com/datasets/zafarali27/car-price-prediction) — 2500 baris, 10 kolom.
- **Fitur (X)**: `Brand`, `Year`, `Engine Size`, `Fuel Type`, `Transmission`, `Mileage`, `Condition`, `Model`
- **Target (y)**: `Price`
- **Algoritma dibandingkan**: Linear Regression vs Random Forest Regressor

## 2. Struktur Folder

```
project/
├── data/
│   └── car_price_prediction.csv     # dataset mentah
├── model/
│   ├── model.pkl                    # pipeline terbaik (preprocessing + model)
│   ├── feature_options.json         # opsi kategori & rentang numerik untuk UI
│   └── model_comparison.csv         # tabel perbandingan metrik evaluasi
├── model_training.ipynb             # notebook lengkap: EDA -> preprocessing -> training -> evaluasi -> simpan model
├── inference.py                     # skrip inferensi standalone
├── app.py                           # aplikasi Streamlit
├── requirements.txt
└── README.md
```

## 3. Menjalankan Secara Lokal

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Opsional) Latih ulang model dari awal
jupyter notebook model_training.ipynb
# atau jalankan langsung:
python train_model.py

# 4. Tes skrip inferensi
python inference.py

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## 4. Ringkasan Preprocessing & Feature Engineering

1. **Data Cleaning**: cek missing value (tidak ada), cek & hapus duplikat (tidak ada), cek & hapus outlier numerik dengan metode IQR (tidak ditemukan outlier signifikan pada dataset ini), drop kolom `Car ID` (hanya identifier, tidak prediktif).
2. **Feature Scaling**: `Year`, `Engine Size`, `Mileage` distandardisasi dengan `StandardScaler`.
3. **Feature Encoding**: `Brand`, `Fuel Type`, `Transmission`, `Condition`, `Model` di-encode dengan `OneHotEncoder`.
4. Kedua langkah di atas dibungkus dalam satu `ColumnTransformer` + `Pipeline` scikit-learn, sehingga:
   - Urutan transformasi saat training dan saat inferensi **selalu identik** (poin penting di rubrik tugas).
   - Cukup 1 file (`model.pkl`) untuk menyimpan model *sekaligus* scaler & encoder — tidak perlu file terpisah.

## 5. Hasil Evaluasi

Lihat tabel lengkap di `model/model_comparison.csv` (dihasilkan otomatis setiap kali notebook dijalankan). Model dengan **R² tertinggi** dan **RMSE/MAE terendah** dipilih sebagai model terbaik dan disimpan sebagai `model/model.pkl`.

> Catatan: dataset ini bersifat sintetis (dibuat untuk latihan), sehingga hubungan antar fitur dan harga tidak selalu kuat/realistis. Fokus tugas adalah **kelengkapan dan kebenaran alur**, bukan akurasi tertinggi.

## 6. Deployment ke Streamlit Community Cloud

1. Push seluruh folder `project/` ini (termasuk folder `model/` yang berisi `model.pkl`) ke sebuah **repository GitHub baru** (public).
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik **"New app"** → pilih repository, branch, dan set **Main file path** ke `app.py`.
4. Klik **Deploy**. Streamlit Cloud otomatis membaca `requirements.txt` dan menginstall dependency.
5. Setelah build selesai, kamu akan dapat tautan publik seperti `https://<nama-app>.streamlit.app`.

### Alternatif: Deploy ke Hugging Face Spaces

1. Buka [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pilih **SDK: Streamlit**, beri nama Space, set visibility Public.
3. Upload semua file (`app.py`, `requirements.txt`, folder `model/`) ke Space tersebut (lewat web UI, drag & drop, atau `git push` — Spaces adalah repo Git).
4. Space otomatis build dan menyediakan tautan publik `https://huggingface.co/spaces/<username>/<nama-space>`.

## 7. Sumber

- Dataset: [Car Price Prediction — Kaggle, oleh zafarali27](https://www.kaggle.com/datasets/zafarali27/car-price-prediction)
- Library: scikit-learn, pandas, numpy, joblib, streamlit
