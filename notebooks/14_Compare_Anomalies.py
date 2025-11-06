
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import os

# Load the results from the three models
df_iso = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_isoforest.csv")
df_lof = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_lof.csv")
df_auto = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane_anomalies_autoencoder.csv")

# Create a summary dataframe
df_summary = pd.DataFrame({
    'IsolationForest': df_iso['Anomaly_IsolationForest'] == -1,
    'LOF': df_lof['Anomaly_LOF'] == -1,
    'Autoencoder': df_auto['Anomaly_Autoencoder'] == 1
})

# --- 1. In-depth Comparison ---

# Calculate the overlap
overlap_all_three = (df_summary['IsolationForest'] & df_summary['LOF'] & df_summary['Autoencoder']).sum()
overlap_iso_lof = (df_summary['IsolationForest'] & df_summary['LOF']).sum()
overlap_iso_auto = (df_summary['IsolationForest'] & df_summary['Autoencoder']).sum()
overlap_lof_auto = (df_summary['LOF'] & df_summary['Autoencoder']).sum()

print("--- Anomaly Detection Overlap ---")
print(f"Anomalies detected by Isolation Forest only: {(df_summary['IsolationForest'] & ~df_summary['LOF'] & ~df_summary['Autoencoder']).sum()}")
print(f"Anomalies detected by LOF only: {(~df_summary['IsolationForest'] & df_summary['LOF'] & ~df_summary['Autoencoder']).sum()}")
print(f"Anomalies detected by Autoencoder only: {(~df_summary['IsolationForest'] & ~df_summary['LOF'] & df_summary['Autoencoder']).sum()}")
print("-" * 20)
print(f"Overlap between Isolation Forest and LOF: {overlap_iso_lof}")
print(f"Overlap between Isolation Forest and Autoencoder: {overlap_iso_auto}")
print(f"Overlap between LOF and Autoencoder: {overlap_lof_auto}")
print("-" * 20)
print(f"Anomalies detected by all three models: {overlap_all_three}")

# Analyze the characteristics of the common anomalies
print("\n--- Characteristics of Anomalies Detected by All Three Models ---")
common_anomalies_indices = df_summary[df_summary.all(axis=1)].index
common_anomalies_df = df_iso.loc[common_anomalies_indices]
print(common_anomalies_df.describe())


# --- 2. Visualizations ---

# Create the static directory if it doesn't exist
static_dir = r"C:\Users\DELL\Desktop\predict\app\static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Venn Diagram
plt.figure(figsize=(10, 10))
venn3(subsets=(df_summary['IsolationForest'].sum() - (overlap_iso_lof + overlap_iso_auto - overlap_all_three),
                  df_summary['LOF'].sum() - (overlap_iso_lof + overlap_lof_auto - overlap_all_three),
                  overlap_iso_lof - overlap_all_three,
                  df_summary['Autoencoder'].sum() - (overlap_iso_auto + overlap_lof_auto - overlap_all_three),
                  overlap_iso_auto - overlap_all_three,
                  overlap_lof_auto - overlap_all_three,
                  overlap_all_three),
      set_labels=('Isolation Forest', 'Local Outlier Factor', 'Autoencoder'))
plt.title("Overlap of Anomaly Detection Models", fontsize=16)
plt.savefig(os.path.join(static_dir, 'brisbane_anomaly_overlap_venn.png'))
plt.close()

# Scatter Plot
plt.figure(figsize=(14, 9))
df_summary['Chlorophyll'] = df_iso['Chlorophyll']
df_summary['Turbidity'] = df_iso['Turbidity']

# Create a category column for plotting
def get_category(row):
    if row['IsolationForest'] and row['LOF'] and row['Autoencoder']:
        return 'All Three'
    if row['IsolationForest'] and row['LOF']:
        return 'ISO & LOF'
    if row['IsolationForest'] and row['Autoencoder']:
        return 'ISO & Auto'
    if row['LOF'] and row['Autoencoder']:
        return 'LOF & Auto'
    if row['IsolationForest']:
        return 'Isolation Forest Only'
    if row['LOF']:
        return 'LOF Only'
    if row['Autoencoder']:
        return 'Autoencoder Only'
    return 'Normal'

df_summary['Category'] = df_summary.apply(get_category, axis=1)

sns.scatterplot(data=df_summary[df_summary['Category'] != 'Normal'], 
                x='Chlorophyll', y='Turbidity', hue='Category', palette='viridis', s=100, alpha=0.7)
plt.title('Anomalies by Chlorophyll and Turbidity', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_anomaly_comparison_scatter.png'))
plt.close()

print("\nComparison and visualization complete. Plots saved to app/static directory.")
