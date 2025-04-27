import streamlit as st
import numpy as np
import librosa
import pickle

# Load saved models
with open('models/genre_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to extract features
def extract_features(file):
    audio, sample_rate = librosa.load(file, duration=30)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Streamlit UI
st.title("🎶 Music Genre Classifier")

st.write("Upload a `.wav` file and predict the genre!")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    features = extract_features(uploaded_file)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted_genre = label_encoder.inverse_transform(prediction)
    
    st.success(f"🎵 Predicted Genre: **{predicted_genre[0]}**")
