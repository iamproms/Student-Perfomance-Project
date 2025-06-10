import pandas as pd
from sklearn.preprocessing import StandardScaler

def map_to_performance_category(gpa):
    if gpa >= 2.5:
        return "Passing"
    elif gpa >= 1.5:
        return "Average"
    else:
        return "Low"


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df_clean = df.drop(columns=['StudentID', 'GradeClass'])  # Remove GradeClass

    # Map GPA to performance category
    df_clean['PerformanceCategory'] = df_clean['GPA'].apply(map_to_performance_category)
    df_clean = df_clean.drop(columns=['GPA'])  # GPA becomes internal — drop after mapping

    X = df_clean.drop(columns=['PerformanceCategory'])
    y = df_clean['PerformanceCategory']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y, scaler
