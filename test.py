import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load("models/best_random_forest_model.pkl")

print(model.feature_importances_)
print(model.feature_importances_.shape)

# Define your feature names — the ones you used during training
feature_names = [
    'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
    'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
    'Sports', 'Music', 'GPA'
]


# Get feature importances
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Print top features
print(importance_df)

# Plot if you like
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

"""

# Feature Correlation Heatmap
This code snippet generates a heatmap to visualize the correlation between features in the dataset.

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Student_performance_data _.csv")

corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
# This code generates a heatmap to visualize the correlation between features in the dataset.
# It uses seaborn and matplotlib to create a heatmap of the correlation matrix.
"""

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load your training dataset (adjust path if needed)
# df = pd.read_csv("Student_performance_data _.csv")

# # Optionally, only select numeric columns for correlation
# numeric_df = df.select_dtypes(include='number')

# # Compute correlation matrix
# corr = numeric_df.corr()

# # Show correlation values
# print(corr)

# # Plot heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
# plt.title("Correlation Matrix of Student Performance Features")
# plt.tight_layout()
# plt.show()
