from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate_model
from visualization import plot_feature_importance, plot_confusion_matrix

if __name__ == '__main__':
    FILE_PATH = "Student_performance_data _.csv"
    X, y = load_and_preprocess_data(FILE_PATH)
    best_model, X_test, y_test = train_and_evaluate_model(X, y)
    plot_feature_importance(best_model, X.columns)
    plot_confusion_matrix(best_model, X_test, y_test)
