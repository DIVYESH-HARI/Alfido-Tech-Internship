# 🚢 Titanic Survival Prediction: A Streamlit ML App

A complete machine learning workflow and interactive web application to predict passenger survival on the RMS Titanic. This project uses **Logistic Regression** and robust data preprocessing to analyze how factors like class, sex, age, and fare influenced a passenger's chance of survival.



[Image of Titanic ship]


## 💡 Project Goal

The primary goal is to build, train, and deploy an accurate classification model that can predict the binary outcome (`Survived`: 0 or 1) for a given Titanic passenger, and to make this prediction accessible via an interactive **Streamlit** application.

---

## 📦 Overview and Methodology

This project follows a standard, robust machine learning pipeline, emphasizing preprocessing to handle the messy real-world data:

### 1. Data Preparation
* **Data Loading & Exploration:** Initial analysis of features, distributions, and the target variable (`Survived`).
* **Feature Engineering:** Extracting titles (Mr., Mrs., Miss, etc.) from the `Name` column for better imputation and predictive power.
* **Robust Missing Value Imputation:**
    * **Categorical:** Use `SimpleImputer` with the **'most_frequent'** strategy for features like `Embarked`.
    * **Numerical:** Use `SimpleImputer` with the **'median'** strategy for features like `Age` and `Fare`.
* **Categorical Encoding:** Applying **`OneHotEncoder`** to convert categorical features (`Sex`, `Embarked`, `Pclass`) into a format suitable for the Logistic Regression model.
* **Feature Scaling:** Applying **`StandardScaler`** to numerical features (`Age`, `Fare`) to normalize their range, which is crucial for gradient-based models like Logistic Regression.

### 2. Model Building & Training
* **Model:** **Logistic Regression** (`scikit-learn`)
* **Pipeline:** All preprocessing steps are combined into a single `Pipeline` to ensure consistency and prevent data leakage during training.
* **Training:** The model is trained on the preprocessed Titanic training data.

### 3. Deployment
* **Web App:** An interactive front-end is built using **Streamlit**, allowing users to input passenger details and receive real-time survival predictions.

---

## 🛠️ Requirements and Installation

To run this project locally, you need Python and the required libraries.

### Prerequisites

You should have Python 3.8+ installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/titanic-survival-prediction.git](https://github.com/your-username/titanic-survival-prediction.git)
    cd titanic-survival-prediction
    ```
2.  **Install dependencies:**
    The project uses the following libraries:
    * `pandas>=1.5.0`
    * `numpy>=1.21.0`
    * `scikit-learn>=1.2.0`
    * `matplotlib>=3.6.0`
    * `seaborn>=0.12.0`
    * **`streamlit>=1.20.0`** (for deployment)
    * `jupyterlab>=3.6.0` (for exploration/notebook)
    * `joblib>=1.2.0` (for model saving)

    Install them all using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run the App

The main application is deployed via Streamlit.

1.  **Ensure all dependencies are installed** (see above).
2.  **Run the Streamlit application** from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  A browser window will automatically open, or you can navigate to the local URL (usually `http://localhost:8501`).

---

## 📊 Evaluation and Results

The model was evaluated on a held-out test set to ensure robust performance.

### Key Metrics
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | [Insert your model's test accuracy here, e.g., **0.815**] | The proportion of correct predictions. |
| **ROC AUC** | [Insert your model's ROC AUC score here, e.g., **0.86**] | A measure of separability; how well the model distinguishes between survivors and non-survivors. |
| **F1-Score** | [Insert your model's F1 score here, e.g., **0.75**] | The harmonic mean of precision and recall. |

### Confusion Matrix
The confusion matrix visually shows the model's performance:



* **True Positives (TP):** Correctly predicted survivors.
* **True Negatives (TN):** Correctly predicted non-survivors.
* **False Positives (FP):** Predicted survival, but the passenger did not survive (Type I error).
* **False Negatives (FN):** Predicted non-survival, but the passenger survived (Type II error).

### Feature Importance (Coefficients)
In Logistic Regression, the coefficients indicate the strength and direction of a feature's impact on the log-odds of survival.

| Feature | Coefficient Value | Impact on Survival |
| :--- | :--- | :--- |
| **Sex\_female** | Large Positive | **Strongest Positive Predictor** (Women were more likely to survive) |
| **Pclass\_1** | Positive | First class passengers had higher odds of survival |
| **Age** | Small Negative | Older passengers had slightly lower odds of survival |
| **Fare** | Small Positive | Higher fares correlate with slightly higher odds |
| **Pclass\_3** | Large Negative | **Strongest Negative Predictor** (Third class passengers were least likely to survive) |

**Conclusion:** The adage *"Women and children first!"* is strongly supported, as **Sex** and **Passenger Class (Pclass)** are the most dominant factors in the model's prediction.

---