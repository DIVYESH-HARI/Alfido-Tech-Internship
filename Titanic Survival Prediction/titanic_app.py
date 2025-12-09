import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🚢 Titanic Survival Predictor")
st.markdown("Logistic Regression model trained on Titanic data.")

# Load pipeline
# The pipeline likely contains a preprocessing step (like a ColumnTransformer or OneHotEncoder)
# followed by the final model (LogisticRegression).
@st.cache_resource
def load_pipeline():
    # Make sure 'titanic_pipeline.pkl' is in the same directory as app.py
    return joblib.load('titanic_pipeline.pkl')

pipeline = load_pipeline()

# --- Input Form ---
pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 28)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.slider("Fare (£)", 0.0, 520.0, 14.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# --- Data Preparation (THE FIX) ---

# 1. Create the input DataFrame
input_df = pd.DataFrame([{
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Pclass': pclass,
    'Sex': sex,
    'Embarked': embarked
}])

# 2. Define the expected feature columns after preprocessing/OHE
# This list MUST match the feature names and order the model was trained on.
# Based on the error, the OHE used drop_first=True, creating these specific dummy columns:
EXPECTED_FEATURES = [
    'Age', 
    'SibSp', 
    'Parch', 
    'Fare', 
    'Pclass_2', 
    'Pclass_3', 
    'Sex_male', 
    'Embarked_Q', 
    'Embarked_S'
]

# 3. Manually apply One-Hot Encoding to match the pipeline's expectation
ohe_input_df = pd.get_dummies(
    input_df, 
    columns=['Sex', 'Pclass', 'Embarked'], 
    drop_first=True # Matches the training step that created Sex_male, Pclass_2, etc.
)

# 4. Align the columns: add missing OHE columns (e.g., if Pclass=1, Pclass_2 and Pclass_3 will be missing)
for col in EXPECTED_FEATURES:
    if col not in ohe_input_df.columns:
        # Create the missing dummy column with a value of 0
        ohe_input_df[col] = 0

# 5. Select and reorder columns to ensure they are EXACTLY as the model expects
final_input_df = ohe_input_df[EXPECTED_FEATURES]


# --- Predict ---
if st.button("🔍 Predict"):
    try:
        # Pass the fully preprocessed DataFrame to the pipeline
        pred = pipeline.predict(final_input_df)[0]
        prob = pipeline.predict_proba(final_input_df)[0]
        
        if pred == 1:
            st.success(f"✅ Likely **SURVIVED** — Probability: {prob[1]:.2%}")
        else:
            st.error(f"❌ Likely **NOT SURVIVED** — Probability: {prob[0]:.2%}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.dataframe(final_input_df) # Optional: show the user the processed input for debugging

st.caption("💡 Tip: Women, children, and 1st-class passengers had higher survival rates.")