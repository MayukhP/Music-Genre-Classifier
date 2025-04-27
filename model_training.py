import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the extracted features
data = pd.read_csv('data/features.csv')

# Separate features (X) and labels (y)
X = data.drop('label', axis=1)
y = data['label']

# Encode the labels (convert genre names to numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features (important for ML models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Save the model, scaler, and label encoder
with open('models/genre_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model, scaler, and label encoder saved to models/")
