# Python AI Core Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)](https://pandas.pydata.org/)
[![Pytest TDD](https://img.shields.io/badge/Testing-Pytest%20TDD-0a9edc.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Bahasa Indonesia](https://img.shields.io/badge/Language-Bahasa%20Indonesia-green.svg)](README_ID.md)

> 📖 **Language**: **English** | [Versi Bahasa Indonesia](README_ID.md)

A rigorous, test-driven collection of production-grade foundational components for Artificial Intelligence and Machine Learning engineering:
1. **Data Structures & Algorithms**: Custom $O(1)$ LRU Cache for token caching and embedding retrieval.
2. **High-Performance NumPy Vectorization**: Matrix cosine similarity, numerically stable Softmax, and Scaled Dot-Product Attention (Vaswani et al.) without Python loops.
3. **Pandas Feature Engineering Pipelines**: Clean imputation strategies, circular periodic time encoding ($\sin / \cos$), and entity-level rolling window feature extraction.

---

## 📌 Repository Purpose

Demonstrates mastery of core computer science fundamentals, vectorization efficiency, and production data pipeline engineering required for senior AI and Machine Learning engineering roles.

---

## 🏗️ Architecture & Modules

```text
python-ai-core-toolkit/
├── requirements.txt                    # Core requirements (numpy, pandas, pytest)
├── run_all_challenges.py               # Automated test runner and progress tracker
├── challenges/
│   ├── c01_data_structures/            # Cache and memory data structures
│   │   ├── lru_cache.py                # O(1) LRU Cache implementation
│   │   └── test_lru_cache.py           # Eviction and ordering test suite
│   ├── c02_numpy_vectorization/        # Mathematical AI kernels
│   │   ├── math_kernels.py             # Vectorized Cosine Sim, Softmax & Attention
│   │   └── test_math_kernels.py        # Numerical stability & shape verification
│   └── c03_pandas_pipeline/            # Feature engineering pipelines
│       ├── transformer.py              # Imputation, Cyclical Encoding & Rolling Aggs
│       └── test_transformer.py         # Immutability and aggregation verification
```

---

## 🚀 Running the Test Suite

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated progress tracker
python run_all_challenges.py

# 3. Run individual module test suites
pytest challenges/c01_data_structures/
pytest challenges/c02_numpy_vectorization/
pytest challenges/c03_pandas_pipeline/
```

---

## 👨‍💻 Author & Credits

- **Author**: **M. Athallah Rafif Aldera, S.Kom.**
- **Alma Mater**: Universitas Sriwijaya (Computer Systems Engineering)
- **License**: MIT License
