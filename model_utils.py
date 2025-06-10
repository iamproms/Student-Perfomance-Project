import joblib
from sklearn.ensemble import RandomForestClassifier

def save_model(model, path='model.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path='model.joblib'):
    return joblib.load(path)

def predict_new_data(model, scaler, data_df):
    scaled = scaler.transform(data_df)
    return model.predict(scaled)
