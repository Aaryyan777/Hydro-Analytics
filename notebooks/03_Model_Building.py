import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\drinking-water-quality-preprocessed.csv")

# Separate features and target
X = df.drop(columns=['Deterioration'])
y = df['Deterioration']

# One-hot encode the 'Sample class' column
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[['Sample class']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Sample class']))
X = pd.concat([X.drop(columns=['Sample class']), X_encoded_df], axis=1)

# Define the model
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model
metrics = ['accuracy', 'f1', 'precision', 'recall']
for metric in metrics:
    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
    print(f"Mean {metric}: {scores.mean():.4f}")

# Train the final model
model.fit(X, y)

# Save the model
joblib.dump(model, 'water_quality_model.joblib')
joblib.dump(encoder, 'encoder.joblib')

print("\nModel training complete. Model saved to water_quality_model.joblib")
