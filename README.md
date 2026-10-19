# 📦 Problem Solving Agent: Optimasi 1D Bin Packing Problem (1D-BPP)

**Mata Kuliah:** AI5103 Fundamental of AI & Data Science  
**Tugas:** Problem Solving Agent — Tugas Kelompok  
**Anggota Kelompok:** Marshaniswah Syamsul, dkk.  
**Repository GitHub:** [https://github.com/marshaniswah/Tugas-Kelompok-Problem-Solving-Agent.git](https://github.com/marshaniswah/Tugas-Kelompok-Problem-Solving-Agent.git)

---

## 📌 Ringkasan Proyek

Repositori ini memuat implementasi lengkap, mandiri, dan dapat direproduksi (*reproducible*) dari agen pemecah masalah untuk permasalahan optimasi kombinatorial **One-Dimensional Bin Packing Problem (1D-BPP)**. Proyek ini membandingkan tiga pendekatan algoritma:

1. **First Fit Decreasing (FFD)** — Heuristik deterministik *greedy* berkecepatan tinggi sebagai *baseline*.
2. **Genetic Algorithm (GA)** — Metaheuristik berbasis permutasi dengan operator khusus *Order Crossover (OX)*, *Swap Mutation*, seleksi *tournament*, dan elitisme.
3. **Particle Swarm Optimization (PSO)** — Metaheuristik kontinu berbasis populasi yang dipetakan ke solusi diskrit menggunakan representasi *Random-Key Heuristic (`np.argsort`)*.

Pengujian dilakukan pada 5 instans benchmark standar internasional **Falkenauer U120** (`Falkenauer_u120_00.txt` hingga `u120_04.txt`) dengan $n=120$ item dan kapasitas bin $C=150$, lengkap dengan perbandingan terhadap **Best Known Solution (BKS)** dan batas bawah teoretis **Continuous Lower Bound ($L_1$)**.

---

## ⚙️ 1. Cara Instalasi

Pastikan Python 3.10 atau versi yang lebih baru telah terpasang. Jalankan perintah berikut untuk menginstal seluruh dependensi:

```bash
# 1. Clone repository (atau ekstrak berkas zip)
git clone https://github.com/marshaniswah/Tugas-Kelompok-Problem-Solving-Agent.git
cd Tugas-Kelompok-Problem-Solving-Agent

# 2. Install dependensi
pip install -r requirements.txt
```

---

## 🚀 2. Cara Menjalankan & Mereproduksi Tabel/Gambar (1 Perintah)

Seluruh tabel, kurva konvergensi, analisis sensitivitas parameter, dan uji signifikansi statistik dapat direproduksi secara langsung dari data mentah benchmark hanya dengan **satu perintah**:

```bash
python run_all.py
```

Atau buka dan jalankan notebook utama langsung di Jupyter / VS Code:
```bash
jupyter notebook code.ipynb
```
*(Seluruh sel di dalam `code.ipynb` telah dirancang **runut dan mandiri (*self-contained*)** tanpa memanggil modul dari folder eksternal, sehingga dapat langsung dieksekusi dari awal hingga akhir via menu **Run All Cells**).*

---

## 🌐 3. Cara Menjalankan Live Demo Interaktif (Streamlit)

Aplikasi demonstrasi interaktif visualizer (tampilan solusi balok bin, kurva konvergensi animasi bergerak, dan komparasi instan) dapat dijalankan dengan perintah:

```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 📁 4. Struktur Berkas & Konfigurasi Parameter

Semua parameter eksperimen dan *random seeds* tersimpan secara terpusat pada berkas konfigurasi:
`config/experiment_config.json`

```text
├── code.ipynb                       # Notebook utama laporan mandiri (Bahasa Indonesia)
├── app.py                           # Aplikasi Live Demo Interaktif (Streamlit)
├── run_all.py                       # Script reproduksi otomatis 1-perintah
├── requirements.txt                 # Daftar pustaka yang dibutuhkan
├── README.md                        # Panduan instalasi dan reproduksi
│
├── config/
│   └── experiment_config.json       # Berkas konfigurasi parameter, seeds, dan target BKS
│
├── data/
│   └── Falkenauer_U/                # Berkas instans benchmark mentah (.txt)
│       ├── Falkenauer_u120_00.txt
│       ├── Falkenauer_u120_01.txt
│       ├── Falkenauer_u120_02.txt
│       ├── Falkenauer_u120_03.txt
│       └── Falkenauer_u120_04.txt
│
└── results/
    ├── raw/                         # Rekaman mentah seluruh run independen
    └── summaries/                   # Tabel agregasi hasil eksperimen
```

---

## 📊 5. Rangkuman Hasil Komparasi Eksperimental

| Algoritma | Rata-rata Bin | Rata-rata Gap vs BKS (%) | Rata-rata Waktu (*Runtime*) | Status Validasi |
| :--- | :---: | :---: | :---: | :---: |
| **BKS (Optimum)** | 48.40 | 0.00% | - | Optimum Global |
| **FFD (Greedy)** | 49.00 | 1.26% | 0.0002 detik ($< 1$ ms) | 100% Valid |
| **Genetic Algorithm (GA)** | 49.56 | 2.39% | 1.1802 detik | 100% Valid |
| **Particle Swarm (PSO)** | 49.84 | 2.98% | 1.2355 detik | 100% Valid |

*(Diuji pada 5 instans benchmark $\times$ 5 seeds = total 50 runs metaheuristik dengan anggaran 3.000 evaluasi berimbang).*
