# Music Genre Classification

This is a Machine Learning project that classifies music into different genres based on their audio features. The model is trained using a Random Forest classifier on audio features extracted using **Librosa**. A **Streamlit web app** is also built to allow users to upload audio files and get the predicted music genre.

## Project Overview

This project involves the following key components:

1. **Data Collection & Preprocessing:**
   - The project uses the **GTZAN Music Genre Dataset** (a collection of 1,000 audio tracks from 10 genres).
   - Audio features are extracted using **Librosa**, focusing on **MFCC (Mel Frequency Cepstral Coefficients)**.

2. **Model Training:**
   - A Random Forest classifier is trained on the extracted features.
   - Model accuracy of approximately **67%** is achieved on the test set.

3. **Web App:**
   - A **Streamlit** web application is built for users to upload audio files and predict the genre.

## Features

- **Feature Extraction:** Uses **MFCC** features to represent audio data.
- **Model Training:** Trained on 10 different genres.
- **Streamlit Web App:** A simple and interactive interface to classify music genres.
- **Accuracy:** Model achieves ~67% accuracy on the test set.

## Project Structure

