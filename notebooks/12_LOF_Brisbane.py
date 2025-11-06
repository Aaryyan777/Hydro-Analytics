
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")

# Drop the rule-based 'Deterioration' column for unsupervised learning
X = df.drop(columns=['Deterioration'])

# Initialize and train the LocalOutlierFactor model
# n_neighbors is a key parameter. We'll start with the default of 20.
# contamination is also used here to define the threshold for outliers.
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
lof.fit(X)

# Predict the anomalies (-1 for anomalies, 1 for inliers)
# Note: For LOF, prediction on the training data is not standard.
# We use the negative_outlier_factor_ attribute and the offset_ to determine anomalies.
y_pred = lof.predict(X)

# Add the predictions to the original dataframe
df['Anomaly_LOF'] = y_pred

# Save the dataframe with anomaly predictions
df.to_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_lof.csv", index=False)

import joblib

# Save the trained model
joblib.dump(lof, 'lof_model.joblib')

# --- Analysis ---
# Count the number of anomalies found
num_anomalies = (y_pred == -1).sum()
print(f"Number of anomalies detected by Local Outlier Factor: {num_anomalies}")

# Display the characteristics of the detected anomalies
print("\n--- Characteristics of Detected Anomalies ---")
anomalies = df[df['Anomaly_LOF'] == -1]
print(anomalies.describe())

print("\n--- Characteristics of Normal Data ---")
normal_data = df[df['Anomaly_LOF'] == 1]
print(normal_data.describe())

print("\nLocal Outlier Factor analysis complete. Results saved to brisbane_anomalies_lof.csv")
