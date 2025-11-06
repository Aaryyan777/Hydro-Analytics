
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")

# Separate features and target
X = df.drop(columns=['Deterioration'])
y = df['Deterioration']

# Use SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Define the model
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model
metrics = ['accuracy', 'f1', 'precision', 'recall']
for metric in metrics:
    scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring=metric)
    print(f"Mean {metric}: {scores.mean():.4f}")

# Train the final model
model.fit(X_resampled, y_resampled)

# Save the model
joblib.dump(model, 'water_quality_model_brisbane_xgb.joblib')

print("\nModel training complete. Model saved to water_quality_model_brisbane_xgb.joblib")
