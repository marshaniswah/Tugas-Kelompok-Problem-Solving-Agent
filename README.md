# 📦 Problem Solving Agent: Optimasi 1D Bin Packing Problem (1D-BPP)

**Mata Kuliah:** AI5103 Fundamental of AI & Data Science  
**Tugas:** Problem Solving Agent — Tugas Kelompok  
**Anggota Kelompok:** Marshaniswah Syamsul, Hasna Putri Priswati.  
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
*(Seluruh sel di dalam `code.ipynb` telah dibuat **runut dan mandiri (*self-contained*)** tanpa memanggil modul dari folder eksternal, sehingga dapat langsung dieksekusi dari awal hingga akhir via menu **Run All**).*

---

## 📁 3. Struktur Berkas & Konfigurasi Parameter

Semua parameter eksperimen dan *random seeds* tersimpan secara terpusat pada berkas konfigurasi:
`config/experiment_config.json`

```text
├── code.ipynb                       # Notebook utama laporan mandiri (Bahasa Indonesia)
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

```

---
