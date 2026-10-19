"""
Script Eksekusi Satu Perintah (One-Command Reproducibility Runner)
AI5103 Fundamental of AI & Data Science — Problem Solving Agent
Kelompok: Marshaniswah Syamsul, dkk.

Menjalankan dan mereproduksi seluruh eksperimen, tabel, dan grafik
dari data mentah instans Falkenauer U120.
"""

import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    notebook_path = root / "bpp_experiments_indonesia.ipynb"
    
    print("=" * 70)
    print("🚀 MEMULAI EKSEKUSI REPRODUKSI 1D BIN PACKING PROBLEM (1D-BPP)")
    print("=" * 70)
    print(f"File Notebook Target: {notebook_path.name}")
    print("Mengeksekusi seluruh sel secara langsung dari data benchmark...")
    
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", str(notebook_path),
        "--output", str(notebook_path)
    ]
    
    res = subprocess.run(cmd, cwd=str(root))
    if res.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ SUKSES: Seluruh tabel dan grafik berhasil direproduksi!")
        print(f"Hasil tersimpan di: {notebook_path}")
        print("=" * 70)
    else:
        print("\n❌ Terjadi kesalahan saat eksekusi nbconvert. Menjalankan fallback script...")
        fallback_cmd = [sys.executable, str(root / "scripts" / "run_experiments.py")]
        subprocess.run(fallback_cmd, cwd=str(root))

if __name__ == "__main__":
    main()
