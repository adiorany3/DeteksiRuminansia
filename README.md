# Prediksi Penyakit Ternak — Streamlit

Versi Streamlit berbahasa Indonesia dari notebook Kaggle `fall2fire/lifestock-disease-prediction`.

## Perbaikan dibanding notebook asli

- Antarmuka dan kategori diterjemahkan ke Bahasa Indonesia.
- Pipeline menyimpan rekayasa fitur, encoding, dan model sebagai satu artefak.
- `OneHotEncoder` menggantikan `LabelEncoder`, sehingga kategori nominal tidak dianggap memiliki urutan angka.
- Pembagian data dilakukan sebelum model dilatih dan menggunakan stratifikasi.
- Prediksi menampilkan tiga kemungkinan beserta probabilitas, informasi penyakit, dan panduan penanganan awal yang memerlukan arahan dokter hewan.
- Suhu tubuh dimasukkan dalam Celsius (°C), lalu dikonversi otomatis ke Fahrenheit untuk mempertahankan kompatibilitas dengan dataset dan model tersimpan.
- Gejala direpresentasikan multi-hot sehingga pengguna cukup memilih 1–3 gejala; data latih diaugmentasi dengan versi 1 dan 2 gejala.
- Umur ternak dimasukkan lewat poel gigi (jumlah pasang gigi tetap) yang lebih dikenal peternak.
- Batas suhu tidak lagi memiliki celah antara 102,5 dan 102,6 °F.

## Menjalankan secara lokal

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Unggah semua file proyek, termasuk `model.joblib`, ke repositori GitHub.
2. Buka https://share.streamlit.io dan hubungkan akun GitHub.
3. Pilih repositori dan isi **Main file path** dengan `app.py`.
4. Klik **Deploy**. Aplikasi tidak memerlukan API key atau secret.

## Peringatan

Model ini dibuat dari dataset sintetis/terbatas dan hanya untuk edukasi. Hasilnya bukan diagnosis medis veteriner. Hubungi dokter hewan untuk diagnosis dan tindakan.

Dataset: https://www.kaggle.com/datasets/captaingee/livestock-disease-diagnosis-dataset
