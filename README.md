# Fast-VAT: Accelerated Visual Assessment of Cluster Tendency 🚀

## 📌 Overview

**Fast-VAT** is a high-performance implementation of the Visual Assessment of Cluster Tendency (VAT) algorithm. It allows users to visually assess the presence of clustering structure in datasets before applying clustering algorithms like K-Means or DBSCAN.

This package includes:

* A baseline implementation of VAT using NumPy and SciPy.
* Accelerated variants using **Numba** (JIT compilation) and **Cython** (C-level optimizations).
* Example evaluations and visualizations on standard benchmark datasets.

## ⚡ Features

* ✅ Implements Prim’s-based VAT algorithm for dissimilarity matrix reordering.
* 🚀 Accelerated versions using **Numba** (25–35× faster).
* 🧪 High-performance Cython implementation with up to **50× speedup**.
* 🎯 Validates cluster tendency using **Hopkins statistic**, **PCA**, **t-SNE**.
* 📊 Comparative analysis with **K-Means** and **DBSCAN**.

## 📊 Datasets Used

We evaluated Fast-VAT on the following datasets:

* Iris (3-class flower classification)
* Spotify subset (500×500 feature matrix)
* Mall Customers (customer segmentation)
* Synthetic datasets:

  * Blobs (well-separated clusters)
  * Circles (nonlinear structure)
  * Moons (interleaved crescents)
  * Gaussian Mixture Models (GMM)

> ℹ️ All datasets are preprocessed and stored as `.npy` dissimilarity matrices under the `/data` directory.

## 📥 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Ashx098/VAT-Optimized.git
cd VAT-Optimized
pip install -e .
```


## 🙏 Acknowledgments

This project was developed under the guidance of:

**Prof. Ismael Lachheb**
*EPITA School of Engineering and Computer Science*
Paris, France

**MSR Avinash**
*Presidency University, Bangalore* (Work conducted during exchange at EPITA)

---

