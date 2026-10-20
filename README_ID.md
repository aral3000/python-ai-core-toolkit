# Python AI Core Toolkit (Fondasi Pemrograman & Data AI)

[![English Version](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)](https://pandas.pydata.org/)
[![Pytest TDD](https://img.shields.io/badge/Testing-Pytest%20TDD-0a9edc.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📖 **Bahasa / Language**: [English Version (Main README)](README.md) | **Bahasa Indonesia**

Repositori monorepo berbasis *Test-Driven Development* (TDD) yang memvalidasi penguasaan konsep fundamental *Computer Science* dan *Data Engineering* untuk AI Engineering:
1. **Struktur Data & Algoritma**: Implementasi *LRU Cache* $O(1)$ untuk manajemen *token cache* dan *embedding store*.
2. **NumPy Vektorisasi Berperforma Tinggi**: Matriks kemiripan kosinus (*Cosine Similarity*), Softmax stabil dari *overflow*, dan *Scaled Dot-Product Attention* (mekanisme inti Transformer) tanpa menggunakan loop Python.
3. **Pipeline Data Pandas Produksi**: Imputasi data hilang yang *immutable*, pengkodean fitur periodik siklikal ($\sin/\cos$), serta agregasi fitur *rolling window* per entitas.

---

## 🚀 Panduan Eksekusi

```bash
# Jalankan pelacak progres tantangan
python run_all_challenges.py

# Menjalankan pengujian spesifik per topik
pytest challenges/c01_data_structures/
pytest challenges/c02_numpy_vectorization/
pytest challenges/c03_pandas_pipeline/
```

---

## 👨‍💻 Penulis & Lisensi

- **Penulis**: **M. Athallah Rafif Aldera, S.Kom.**
- **Institusi**: Universitas Sriwijaya (Sistem Komputer)
- **Lisensi**: MIT License
