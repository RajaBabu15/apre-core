import streamlit as st
import os
import numpy as np
import joblib
import xgboost as xgb
import librosa
import matplotlib.pyplot as plt
import sys
import pandas as pd

# Add src to path so we can use your feature extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from apre.features.extractor import extract_advanced_features

# --- CONFIGURATION ---
st.set_page_config(page_title="APRE Demo", page_icon="🎵", layout="wide")

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS (Cached for speed) ---
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("saved_models/scaler.joblib")
        le = joblib.load("saved_models/label_encoder.joblib")
        model = xgb.XGBClassifier()
        model.load_model("saved_models/xgboost_model.json")
        return scaler, le, model
    except FileNotFoundError:
        return None, None, None

scaler, le, model = load_artifacts()

# --- SIDEBAR ---
st.sidebar.title("APRE Settings")
st.sidebar.info("Audio Pattern Recognition Engine")
model_type = st.sidebar.selectbox("Select Model", ["XGBoost (Best: 95%)", "CNN", "MLP"])
st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.write("Detects acoustic events like:")
st.sidebar.code("\n".join(["Chainsaw", "Church Bells", "Clock Alarm", "Dog", "Sea Waves"]))

# --- MAIN CONTENT ---
st.title("🎵 Audio Pattern Recognition Engine")
st.write("Upload an audio file to classify the environmental sound using the trained APRE pipeline.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Audio")
    
    # Tab selection for Upload vs Sample
    tab1, tab2 = st.tabs(["Upload File", "Try Sample"])
    
    audio_file = None
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a WAV file", type=['wav', 'mp3'])
        if uploaded_file is not None:
            audio_file = uploaded_file

    with tab2:
        # List files in 'samples' folder
        sample_files = [f for f in os.listdir("samples") if f.endswith('.wav')] if os.path.exists("samples") else []
        selected_sample = st.selectbox("Select a sample file", sample_files)
        if selected_sample:
            audio_file = os.path.join("samples", selected_sample)

    if audio_file:
        st.audio(audio_file, format='audio/wav')
        
        # Analyze Button
        if st.button("Analyze Audio Pattern"):
            if model is None:
                st.error("Models not found! Please run 'python main.py' first to train and save models.")
            else:
                with st.spinner("Extracting features and classifying..."):
                    # 1. Save temp file if uploaded
                    if hasattr(audio_file, 'read'):
                        with open("temp.wav", "wb") as f:
                            f.write(audio_file.getbuffer())
                        path = "temp.wav"
                    else:
                        path = audio_file

                    # 2. Extract Features (Reuse your existing code)
                    # Note: We need to handle the flattening exactly like training
                    features = extract_advanced_features(path) 
                    
                    # 3. Preprocess
                    # Reshape for single sample (1, -1)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # 4. Predict
                    pred_probs = model.predict_proba(features_scaled)[0]
                    pred_idx = np.argmax(pred_probs)
                    pred_label = le.inverse_transform([pred_idx])[0]
                    confidence = pred_probs[pred_idx]

                    # --- RESULTS COLUMN ---
                    with col2:
                        st.subheader("2. Analysis Results")
                        
                        # Big Prediction Display
                        color = "green" if confidence > 0.8 else "orange"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style='margin:0'>Predicted Class</h3>
                            <h1 style='color:{color}; font-size: 48px; margin:0'>{pred_label.replace('_', ' ').title()}</h1>
                            <p style='font-size: 18px'>Confidence: <b>{confidence*100:.1f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("### Class Probabilities")
                        # Bar Chart of probabilities
                        class_names = le.classes_
                        chart_data = pd.DataFrame(
                            {"Class": class_names, "Probability": pred_probs}
                        )
                        st.bar_chart(chart_data.set_index("Class"))

                        # Optional: Feature Visualization (Waveform)
                        st.write("### Signal Visualization")
                        y, sr = librosa.load(path, sr=22050)
                        fig, ax = plt.subplots(figsize=(10, 3))
                        librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')
                        st.pyplot(fig)
