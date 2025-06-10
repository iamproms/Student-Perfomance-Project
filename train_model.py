import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from data_preprocessing import load_and_preprocess_data
from data_preprocessing import load_and_preprocess_data

# Load and preprocess the data directly

X, y, scaler = load_and_preprocess_data("Student_performance_data _.csv")

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the model and scaler
joblib.dump(model, "models/best_random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model saved to models/best_random_forest_model.pkl")
print("✅ Model saved to models/scaler.pkl")

