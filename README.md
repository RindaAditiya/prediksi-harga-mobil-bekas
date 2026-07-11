# Tugas Akhir — Fundamental Sains Data
## Implementasi Supervised Learning dan Unsupervised Learning

Repository ini adalah tugas akhir mata kuliah **Fundamental Sains Data**. Sesuai ketentuan tugas, di dalamnya ada **dua proyek analisis data yang berbeda dan berdiri sendiri-sendiri**: satu proyek pakai pendekatan *supervised learning* (data punya label/target yang mau diprediksi), satu lagi pakai pendekatan *unsupervised learning* (data tidak punya label, kita cari pola/kelompok tersembunyi di dalamnya). Kedua proyek dikerjakan dengan dataset yang berbeda dan sama sekali tidak saling berhubungan — digabung di satu repo ini hanya karena ketentuan pengumpulan tugasnya minta begitu.

---

## Studi Kasus 1 — Supervised Learning
### Prediksi Harga Mobil Bekas

**Apa yang dikerjakan di sini?**
Proyek ini mencoba menjawab pertanyaan: *"kalau tahu spesifikasi sebuah mobil bekas (merek, tahun, jarak tempuh, dst), berapa kira-kira harga jualnya?"*. Ini termasuk masalah **regresi**, karena yang diprediksi adalah angka (harga), bukan kategori.

**Datanya dari mana?**
Dataset [Car Price Prediction](https://www.kaggle.com/datasets/zafarali27/car-price-prediction) dari Kaggle, dibuat oleh akun bernama zafarali27. Isinya 2.500 baris data mobil dengan 10 kolom informasi.

**Fitur apa saja yang dipakai untuk memprediksi?**
Merek mobil (`Brand`), tahun produksi (`Year`), ukuran mesin (`Engine Size`), jenis bahan bakar (`Fuel Type`), jenis transmisi (`Transmission`), jarak tempuh (`Mileage`), kondisi mobil (`Condition`), dan model mobil (`Model`). Yang mau diprediksi (target) adalah `Price`, alias harga jualnya.

**Algoritma apa yang dibandingkan?**
Dua algoritma dicoba dan dibandingkan performanya: **Linear Regression** (model paling sederhana, menarik garis lurus antar hubungan fitur dan harga) dan **Random Forest Regressor** (model yang lebih kompleks, menggabungkan banyak decision tree untuk hasil yang biasanya lebih akurat).

**Bagaimana cara mengukur model ini bagus atau tidak?**
Pakai tiga metrik: **MAE** (rata-rata seberapa jauh tebakan model dari harga asli, dalam satuan yang sama dengan harga), **RMSE** (mirip MAE tapi lebih menghukum tebakan yang sangat meleset), dan **R² Score** (seberapa besar persentase variasi harga yang berhasil dijelaskan oleh model, dari 0 sampai 1 — semakin dekat ke 1 semakin bagus).

**Di mana kodenya?**
Seluruh proses (mulai dari baca data, eksplorasi data, membersihkan data, melatih model, sampai mengevaluasi hasil) ada di file [`model_training.ipynb`](./model_training.ipynb). Ini file notebook yang bisa dibuka dan dijalankan langkah demi langkah, sambil membaca penjelasan di tiap bagiannya.

**Ada aplikasinya juga?**
Ya, file [`app.py`](./app.py) adalah aplikasi interaktif berbasis Gradio, jadi orang lain bisa mengisi spesifikasi mobil lewat tampilan web sederhana dan langsung dapat prediksi harganya, tanpa perlu tahu coding sama sekali.

**Cara menjalankan notebooknya sendiri:**
```bash
pip install -r requirements.txt
jupyter notebook model_training.ipynb
```
Baris pertama menginstall semua library Python yang dibutuhkan (pandas, scikit-learn, dst). Baris kedua membuka notebook-nya di Jupyter.

**Cara menjalankan aplikasinya:**
```bash
python app.py
```
Setelah dijalankan, akan muncul link lokal (biasanya `http://127.0.0.1:7860`) yang bisa dibuka di browser.

---

## Studi Kasus 2 — Unsupervised Learning
### Segmentasi Pasien Berdasarkan Indikator Kesehatan Diabetes

**Apa yang dikerjakan di sini?**
Beda dengan studi kasus 1, di sini kita tidak punya "jawaban benar" untuk diprediksi. Yang dilakukan adalah mengelompokkan pasien-pasien ke dalam beberapa "profil kesehatan" berdasarkan kemiripan data medis mereka (kadar gula darah, BMI, usia, dst), tanpa memberi tahu model siapa yang sebenarnya kena diabetes dan siapa yang tidak. Tujuannya melihat apakah ada pola alami yang muncul dari data itu sendiri — ini yang disebut **clustering**.

**Datanya dari mana?**
Dataset [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), aslinya dikumpulkan oleh National Institute of Diabetes and Digestive and Kidney Diseases. Berisi data 768 pasien perempuan dengan 8 indikator kesehatan.

**Fitur apa saja yang dipakai untuk mengelompokkan?**
Jumlah kehamilan (`Pregnancies`), kadar glukosa (`Glucose`), tekanan darah (`BloodPressure`), ketebalan lipatan kulit (`SkinThickness`), kadar insulin (`Insulin`), indeks massa tubuh (`BMI`), skor riwayat genetik diabetes (`DiabetesPedigreeFunction`), dan usia (`Age`). Ada juga kolom `Outcome` (status diabetes pasien yang sebenarnya) di dataset ini, tapi kolom itu **sengaja tidak dipakai** saat proses pengelompokan — baru dipakai di akhir untuk mengecek apakah hasil pengelompokan otomatis tadi ternyata masuk akal secara medis atau tidak.

**Algoritma apa yang dipakai?**
Dua metode clustering dicoba dan dibandingkan: **K-Means** (mengelompokkan data berdasarkan titik pusat terdekat) dan **Hierarchical Clustering** (mengelompokkan data secara bertahap, dari yang paling mirip digabung dulu). **PCA** juga dipakai, bukan untuk mengelompokkan, tapi supaya hasil pengelompokan yang tadinya berdimensi 8 (8 fitur) bisa digambar dalam bentuk grafik 2 dimensi yang enak dilihat.

**Bagaimana cara tahu jumlah kelompok yang tepat?**
Tidak ditebak asal, tapi diukur pakai tiga cara: **Elbow Method** (grafik yang menunjukkan titik di mana penambahan kelompok baru sudah tidak banyak membantu lagi), **Silhouette Score** (mengukur seberapa rapat anggota dalam satu kelompok dan seberapa jauh dari kelompok lain), dan **Davies-Bouldin Index** (metrik serupa, tapi arah baiknya kebalikan — semakin kecil semakin bagus).

**Di mana kodenya?**
Semua proses ada di file [`unsupervised_diabetes/unsupervised_diabetes_clustering.ipynb`](./unsupervised_diabetes/unsupervised_diabetes_clustering.ipynb), lengkap dari eksplorasi data sampai kesimpulan hasil pengelompokan.

**Cara menjalankan notebooknya sendiri:**
```bash
cd unsupervised_diabetes
jupyter notebook unsupervised_diabetes_clustering.ipynb
```
Baris pertama masuk ke folder studi kasus 2. Baris kedua membuka notebook-nya. Pastikan file `diabetes.csv` ada di folder yang sama, karena notebook membaca data dari situ.

---

## Isi Folder Repository Ini

Supaya nggak bingung file mana untuk apa, berikut penjelasan tiap folder dan file:

- **`data/`** — berisi file dataset mentah untuk studi kasus 1 (harga mobil), sebelum diolah.
- **`model/`** — berisi model yang sudah selesai dilatih (`model.pkl`), file bantu untuk aplikasi Gradio (`feature_options.json`), dan tabel perbandingan hasil evaluasi model (`model_comparison.csv`).
- **`unsupervised_diabetes/`** — folder khusus untuk seluruh isi studi kasus 2, berisi notebook clustering dan dataset diabetes-nya sendiri.
- **`model_training.ipynb`** — notebook lengkap studi kasus 1 (proses melatih model prediksi harga mobil).
- **`app.py`** — aplikasi Gradio untuk studi kasus 1, supaya model bisa dicoba lewat tampilan web.
- **`inference.py`** — skrip Python sederhana untuk memakai model yang sudah dilatih tanpa perlu membuka notebook atau aplikasi web.
- **`requirements.txt`** — daftar library Python yang dibutuhkan supaya semua kode di repo ini bisa jalan.
- **`README.md`** — file yang sedang kamu baca ini, isinya penjelasan keseluruhan proyek.

---

## Tentang

Repository ini dibuat sebagai tugas akhir mata kuliah Fundamental Sains Data oleh  kelompok manut.
