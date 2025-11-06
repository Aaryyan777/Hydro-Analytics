
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")
X = df.drop(columns=['Deterioration'])

# Load the three trained models
iso_forest = joblib.load('isolation_forest_model.joblib')
lof = joblib.load('lof_model.joblib')
autoencoder = load_model('autoencoder_model.h5')

# --- Get Predictions from Each Model ---

# 1. Isolation Forest
pred_iso = iso_forest.predict(X)

# 2. Local Outlier Factor
pred_lof = lof.predict(X)

# 3. Autoencoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
reconstruction_error = np.mean(np.power(X_scaled - autoencoder.predict(X_scaled), 2), axis=1)
threshold = np.quantile(reconstruction_error, 0.99)
pred_auto = np.where(reconstruction_error > threshold, -1, 1)

# --- Create the Ensemble Prediction (Majority Vote) ---

# Combine the predictions into a single dataframe
predictions_df = pd.DataFrame({
    'IsolationForest': pred_iso,
    'LOF': pred_lof,
    'Autoencoder': pred_auto
})

# A prediction of -1 is an anomaly. Convert to 0 for normal, 1 for anomaly.
predictions_df = (predictions_df == -1).astype(int)

# Sum the votes for each row. If the sum is >= 2, it's an ensemble anomaly.
df['Anomaly_Ensemble'] = (predictions_df.sum(axis=1) >= 2).astype(int)

# Save the dataframe with ensemble predictions
df.to_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_ensemble.csv", index=False)

# --- Analysis ---
# Count the number of anomalies found
num_anomalies = df['Anomaly_Ensemble'].sum()
print(f"Number of anomalies detected by Ensemble (Majority Vote): {num_anomalies}")

# Display the characteristics of the detected anomalies
print("\n--- Characteristics of Detected Anomalies (Ensemble) ---")
anomalies = df[df['Anomaly_Ensemble'] == 1]
print(anomalies.describe())

print("\n--- Characteristics of Normal Data (Ensemble) ---")
normal_data = df[df['Anomaly_Ensemble'] == 0]
print(normal_data.describe())

print("\nEnsemble model analysis complete. Results saved to brisbane_anomalies_ensemble.csv")
