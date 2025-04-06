import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load the landmarks CSV
df = pd.read_csv('data/gestures.csv', header=None)

# Features and labels
X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values   # Last column (Arabic letters)

# Encode Arabic labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a classifier (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved.")