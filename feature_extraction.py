import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=30)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_dataset(dataset_path):
    genres = os.listdir(dataset_path)
    features = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue

        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(genre)

    feature_df = pd.DataFrame(features)
    feature_df['label'] = labels
    return feature_df

if __name__ == "__main__":
    dataset_path = "data/genres"  # folder path where your genres are
    df = prepare_dataset(dataset_path)
    df.to_csv("data/features.csv", index=False)
    print("✅ Feature extraction completed and saved to data/features.csv")
