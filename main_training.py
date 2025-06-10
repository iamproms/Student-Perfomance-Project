# main_training.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import save_model
from evaluation_utils import evaluate_model, print_evaluation
from visualization import plot_feature_importance, plot_confusion_matrix
from logger_config import setup_logger

logger = setup_logger()

# Load dataset
file_path = "Student_performance_data _.csv"
df = pd.read_csv(file_path)

# Drop ID and convert target
df_clean = df.drop(columns=['StudentID'])
df_clean['GradeClass'] = df_clean['GradeClass'].astype(int)

# Separate features and target
X = df_clean.drop(columns=['GradeClass'])
y = df_clean['GradeClass']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluation
logger.info("Evaluating base model")
y_pred = model.predict(X_test)
cm, report_df = evaluate_model(y_test, y_pred)
print_evaluation(cm, report_df)

# Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

logger.info(f"Best Parameters: {grid_search.best_params_}")

# Evaluate tuned model
y_pred_tuned = best_model.predict(X_test)
cm_tuned, report_df_tuned = evaluate_model(y_test, y_pred_tuned)
print_evaluation(cm_tuned, report_df_tuned)

# Save model
save_model(best_model, 'rf_model.joblib')

# Visualizations
plot_feature_importance(best_model, X_scaled_df.columns)
plot_confusion_matrix(cm_tuned, title="Confusion Matrix - Tuned Model")
