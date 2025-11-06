
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")

# Drop the rule-based 'Deterioration' column for unsupervised learning
X = df.drop(columns=['Deterioration'])

# Initialize and train the IsolationForest model
# The 'contamination' parameter is an estimate of the proportion of outliers in the data.
# We'll start with 1%, which is similar to the proportion from our rule-based approach.
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict the anomalies (-1 for anomalies, 1 for inliers)
predictions = iso_forest.predict(X)

# Add the predictions to the original dataframe
df['Anomaly_IsolationForest'] = predictions

# Save the dataframe with anomaly predictions
df.to_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_isoforest.csv", index=False)

# Save the trained model
joblib.dump(iso_forest, 'isolation_forest_model.joblib')

# --- Analysis ---
# Count the number of anomalies found
num_anomalies = (predictions == -1).sum()
print(f"Number of anomalies detected by Isolation Forest: {num_anomalies}")

# Display the characteristics of the detected anomalies
print("\n--- Characteristics of Detected Anomalies ---")
anomalies = df[df['Anomaly_IsolationForest'] == -1]
print(anomalies.describe())

print("\n--- Characteristics of Normal Data ---")
normal_data = df[df['Anomaly_IsolationForest'] == 1]
print(normal_data.describe())

print("\nIsolation Forest analysis complete. Results saved to brisbane_anomalies_isoforest.csv")
