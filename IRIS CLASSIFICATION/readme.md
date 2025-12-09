# 🌸 Iris Flower Species Classifier  
### *A Machine Learning Web App Built with Streamlit & scikit-learn*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-success)
![Last Commit](https://img.shields.io/github/last-commit/your-username/iris-classifier?color=blue)

> *"The Iris dataset is the 'Hello World' of machine learning — and this project brings it to life with an interactive, production-ready interface."*

A **fully functional web application** that classifies Iris flowers into one of three species using morphological measurements. Trained on Fisher’s classic 1936 dataset, this project demonstrates end-to-end ML: data loading → preprocessing → modeling → evaluation → deployment.

✅ **100% test accuracy** | 🚀 Deployable in <2 minutes | 📊 Transparent & interpretable

---

## 📌 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🧠 Model & Evaluation](#-model--evaluation)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🖥️ Web App Features](#️-web-app-features)
- [🔧 Local Setup](#-local-setup)
- [☁️ Deployment](#️-deployment)
- [🧪 Testing & Validation](#-testing--validation)
- [📈 Future Enhancements](#-future-enhancements)
- [📜 License](#-license)
- [📚 References](#-references)
- [📬 Contact](#-contact)

---

## 🎯 Project Overview

This project implements a **supervised classification pipeline** for the Iris dataset using:
- **Algorithm**: Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)
- **Framework**: Python + scikit-learn + pandas
- **Frontend**: Streamlit (interactive web UI)
- **Goal**: Predict species from 4 continuous features

It serves as a **minimal, reproducible template** for ML projects — from Jupyter experimentation to deployed web app.

### ✅ Key Highlights
| Feature | Description |
|--------|-------------|
| **Accuracy** | 100% on held-out test set (30 samples) |
| **Interpretability** | Decision rules visualizable (e.g., `petal_width ≤ 0.8 → Iris-setosa`) |
| **Reproducibility** | Fixed `random_state=42` for all stochastic steps |
| **User-Friendly** | No ML knowledge needed — just input measurements! |
| **Lightweight** | < 50 lines of core modeling code |

---

## 📊 Dataset

The **Iris dataset** (R.A. Fisher, 1936) is a multivariate dataset introduced in *"The use of multiple measurements in taxonomic problems"*.

| Attribute | Value |
|----------|-------|
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris) |
| **Samples** | 150 (50 per class) |
| **Features** | 4 numeric, continuous |
| **Classes** | `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` |
| **File Format** | CSV (`IRIS.csv`) |

### Feature Descriptions
| Feature | Unit | Range (min–max) | Biological Meaning |
|--------|------|-----------------|--------------------|
| `sepal_length` | cm | 4.3 – 7.9 | Length of the outer protective petal-like structure |
| `sepal_width` | cm | 2.0 – 4.4 | Width of the sepal |
| `petal_length` | cm | 1.0 – 6.9 | Length of the inner colorful petals |
| `petal_width` | cm | 0.1 – 2.5 | Width of the petal |

### Class Distribution
| Species | Count | % |
|---------|-------|----|
| `Iris-setosa` | 50 | 33.3% |
| `Iris-versicolor` | 50 | 33.3% |
| `Iris-virginica` | 50 | 33.3% |
| **Total** | **150** | **100%** |

✅ **Perfectly balanced** → no need for resampling or class weighting.

---

## 🧠 Model & Evaluation

### 🔧 Modeling Pipeline
```mermaid
flowchart LR
A[Load Data] --> B[Train-Test Split\n80/20, stratified]
B --> C[Train Decision Tree\nrandom_state=42]
C --> D[Predict on Test Set]
D --> E[Evaluate: Accuracy, Report, Confusion Matrix]