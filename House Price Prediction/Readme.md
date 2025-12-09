# 🏠 House Price Prediction 

A Machine Learning project to predict house prices in Washington State using **Linear Regression**. Built with **Python** and **Streamlit** — with **dual-currency support (USD & INR)**.

---

## 🚀 Quick Start

1.  **Install dependencies**
    ```bash
    conda create -n houseprice python=3.10 -y
    conda activate houseprice
    pip install -r requirements.txt
    ```
2.  **Train the model**
    * Run all cells in `HousePrice_Prediction.ipynb` → generates `house_price_model.pkl`.
3.  **Launch the app**
    ```bash
    streamlit run app.py
    ```
    → Opens interactive predictor at `http://localhost:8501`

---

## 📁 Files Included

* `data.csv` — Housing dataset (date, price, bedrooms, sqft_living, etc.)
* `HousePrice_Prediction.ipynb` — EDA, preprocessing, and Linear Regression training
* `app.py` — Streamlit web app (predicts in USD 💵 and INR 🇮🇳)
* `house_price_model.pkl` — Trained model (auto-generated)
* `model_features.pkl` — Feature list for inference
* `requirements.txt` — Python dependencies
* `README.md` — This file

---

## 💡 Key Features

* ✅ **Predicts house price** using 14+ features (sqft, bedrooms, year built, etc.)
* ✅ **Shows results in USD ($) and INR (₹)** — e.g., $612,500 → ₹5.11 Cr
* ✅ Handles `yr_renovated = 0` → uses `yr_built` during training
* ✅ **Clean, beginner-friendly code** — ideal for learning ML workflow
* ✅ **Works offline** (fixed exchange rate: $1 = ₹83.5)

---

## 🛠️ Sample Prediction (from dataset)

| Feature | Value |
| :--- | :--- |
| **Bedrooms** | 4 |
| **Bathrooms** | 2.5 |
| **Sqft Living** | 2,730 |
| **Year Built** | 1991 |
| **Predicted Price** | **$612,500 (USD) → ₹5.11 Crore (INR)** |
| *(Actual price in dataset: $612,500 — perfect match in this case!)* |

---

## 📜 License

**MIT** — Free to use for education, portfolio, or personal projects.


© 2025 — Built with **Python, scikit-learn & Streamlit**
