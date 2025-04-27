import numpy as np
import librosa
import pickle

# Load the saved model, scaler, and label encoder
with open('models/genre_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_genre(file_path):
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted_genre = label_encoder.inverse_transform(prediction)
    return predicted_genre[0]

if __name__ == "__main__":
    file_path = input("🎵 Enter path to the audio file (.wav): ")
    genre = predict_genre(file_path)
    print(f"🎶 Predicted Genre: {genre}")
