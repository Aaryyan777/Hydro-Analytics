
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")

# Drop the rule-based 'Deterioration' column for unsupervised learning
X = df.drop(columns=['Deterioration'])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the Autoencoder model
input_dim = X_scaled.shape[1]
encoding_dim = 8  # Size of the bottleneck

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)

# Calculate reconstruction error
predictions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
df['Reconstruction_Error'] = mse

# Define a threshold for anomalies (e.g., 99th percentile of the error)
threshold = np.quantile(mse, 0.99)
df['Anomaly_Autoencoder'] = (mse > threshold).astype(int)

# Save the dataframe with anomaly predictions
df.to_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_autoencoder.csv", index=False)

# Save the trained model
autoencoder.save('autoencoder_model.h5')

# --- Analysis ---
# Count the number of anomalies found
num_anomalies = (df['Anomaly_Autoencoder'] == 1).sum()
print(f"Number of anomalies detected by Autoencoder: {num_anomalies}")

# Display the characteristics of the detected anomalies
print("\n--- Characteristics of Detected Anomalies ---")
anomalies = df[df['Anomaly_Autoencoder'] == 1]
print(anomalies.describe())

print("\n--- Characteristics of Normal Data ---")
normal_data = df[df['Anomaly_Autoencoder'] == 0]
print(normal_data.describe())

print("\nAutoencoder analysis complete. Results saved to brisbane_anomalies_autoencoder.csv")
