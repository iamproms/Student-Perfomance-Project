from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    report_df = pd.DataFrame(report).transpose()
    return cm, report_df

def print_evaluation(cm, report_df):
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_df)
