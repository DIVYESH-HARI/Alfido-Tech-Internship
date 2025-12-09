import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ---------------------------
# 🎨 Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------------------
# 🧠 Load the trained model — Pure function (for caching)
# ---------------------------
# !!! IMPORTANT: Update these paths if your files are in a different location
MODEL_PATH = r"C:\Users\divye\iris_decision_tree.pkl"
DATA_PATH = r"C:\Users\divye\Downloads\IRIS.csv" # Path for the sample data CSV

@st.cache_resource
def load_model():
    """Pure function: only loads model — NO Streamlit calls."""
    if not os.path.exists(MODEL_PATH):
        # Return a clear signal that the file was not found
        return None, f"Model file not found at: {MODEL_PATH}"
    try:
        # joblib.load is the only action taken inside the cached function
        model = joblib.load(MODEL_PATH)
        return model, None  # success: return model and no error
    except Exception as e:
        # Return the error message
        return None, str(e)

# Load model — get result + optional error
model, error = load_model()

# ---------------------------
# 🚨 Handle errors OUTSIDE cached function
# ---------------------------
if error:
    # Stop execution if model loading failed
    st.error(f"❌ Failed to load model:\n\n`{error}`")
    st.markdown("Please ensure the model file path is correct.")
    st.stop()
elif model is None:
    # Stop execution if model object is unexpectedly None
    st.error("❌ Model is None. Check loading logic.")
    st.stop()
# THE FIX IS HERE: The st.toast() call was removed to resolve the caching error.

# ---------------------------
# 🌸 App Header
# ---------------------------
st.title("🌸 Iris Flower Species Classifier")
st.markdown("""
Predict the species of an Iris flower based on its measurements.
Trained on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) with a **Decision Tree**.
""")

st.divider()

# ---------------------------
# 📏 Input Form
# ---------------------------
st.subheader("🔧 Enter Flower Measurements (in cm)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=5.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# Prepare the input data for the model
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

st.divider()

# ---------------------------
# 🚀 Predict Button
# ---------------------------
if st.button("🔬 Predict Species", type="primary", use_container_width=True):
    try:
        # Perform Prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        class_names = model.classes_

        # Define Emojis for Display
        species_emojis = {
            "Iris-setosa": "🌸",
            "Iris-versicolor": "🌺",
            "Iris-virginica": "🌼"
        }
        emoji = species_emojis.get(prediction, "🌿")

        st.subheader("✨ Prediction Result")
        
        # Display the result using HTML for better styling
        st.markdown(f"""
        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px;text-align:center;border-left:5px solid #4caf50;">
            <h2>{emoji} {prediction}</h2>
            <p style="font-size:1.2em;color:#2e7d32;">Confidence: <b>{max(probabilities):.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Display Probabilities Table
        st.write("### Class Probabilities:")
        prob_df = pd.DataFrame({
            "Species": class_names,
            "Confidence": probabilities
        }).sort_values("Confidence", ascending=False)
        # Format the confidence column as percentages
        prob_df["Confidence"] = prob_df["Confidence"].apply(lambda x: f"{x:.2%}")
        st.table(prob_df)

    except Exception as e:
        st.exception(f"Prediction failed: {e}")

# ---------------------------
# 📊 Sample Data
# ---------------------------
with st.expander("📊 View Sample Data"):
    try:
        # Load sample data from the specified path
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.head(10), use_container_width=True)
    except FileNotFoundError:
        st.warning(f"⚠ Could not load sample data. File not found at: `{DATA_PATH}`")
    except Exception as e:
        st.warning(f"⚠ Could not load sample data: {e}")

st.divider()
st.caption("Built with ❤️ using Streamlit | Dataset: Iris (Fisher, 1936)")