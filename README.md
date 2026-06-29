# 🧲 Superconductor Critical Temperature Predictor

Aplikasi Streamlit untuk eksplorasi data dan prediksi **suhu kritis (critical temperature)** material superkonduktor menggunakan Machine Learning.

## 📦 Dataset

Dataset `superconduct_train.csv` berisi **21.263 sampel** material superkonduktor dengan **81 fitur fisikokimia** (atomic mass, electron affinity, thermal conductivity, dll.) dan target variabel `critical_temp` dalam Kelvin.

## 🚀 Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy ke Streamlit Cloud

1. Fork/upload repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New App** → pilih repo ini → pilih `app.py`
4. Klik **Deploy**

## 📊 Fitur Aplikasi

| Halaman | Deskripsi |
|---|---|
| 📊 Eksplorasi Data | Preview, statistik deskriptif, dan cek missing values |
| 📈 Visualisasi | Distribusi target, korelasi fitur, scatter plot, box plot |
| 🤖 Prediksi ML | Latih & evaluasi model (Random Forest, Gradient Boosting, Ridge, Linear Regression) |
| 🔍 Prediksi Manual | Input nilai fitur secara manual untuk prediksi suhu kritis |

## 🛠️ Teknologi

- Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn
