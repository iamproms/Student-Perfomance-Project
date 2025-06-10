import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Student_performance_data _.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Drop 'StudentID' as it's just an identifier
df_clean = df.drop(columns=['StudentID'])

# Ensure 'GradeClass' is integer for classification
df_clean['GradeClass'] = df_clean['GradeClass'].astype(int)

# Separate features and target
X = df_clean.drop(columns=['GradeClass'])
y = df_clean['GradeClass']

# Scale numeric features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back into DataFrame for readability
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Optional: preview the preprocessed data
print("Scaled Feature Sample:")
print(X_scaled_df.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# STEP 1: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# STEP 2: Train a Random Forest Classifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# STEP 3: Predict on test data
y_pred = model.predict(X_test)

# STEP 4: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=1,
    scoring='accuracy'
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Show best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\nConfusion Matrix (Tuned Model):")
print(confusion_matrix(y_test, y_pred_tuned))

print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned))



import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the trained model
importances = model.feature_importances_
feature_names = X_scaled_df.columns

# Create a DataFrame for easy plotting
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('Feature Importance - Random Forest Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Tuned Model")
plt.show()

