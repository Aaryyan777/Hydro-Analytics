
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane_water_quality.csv")

# 1. Drop Quality Columns
quality_cols = [col for col in df.columns if '[quality]' in col]
df = df.drop(columns=quality_cols)

# 2. Handle Missing Values (Mean Imputation)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)

# 3. Engineer Target Variable ('Deterioration')
# Define thresholds for deterioration
turbidity_threshold = df['Turbidity'].quantile(0.90)
do_threshold = df['Dissolved Oxygen'].quantile(0.10)
ph_low_threshold = 6.5
ph_high_threshold = 8.5

def is_deteriorated(row):
    if (row['Turbidity'] > turbidity_threshold and 
        (row['Dissolved Oxygen'] < do_threshold or 
         row['pH'] < ph_low_threshold or 
         row['pH'] > ph_high_threshold)):
        return 1
    return 0

df['Deterioration'] = df.apply(is_deteriorated, axis=1)

print("Value counts of Deterioration:")
print(df['Deterioration'].value_counts())

# 4. Feature Engineering (Date/Time)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Sample_Month'] = df['Timestamp'].dt.month
df['Sample_DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Sample_Hour'] = df['Timestamp'].dt.hour

# 5. Drop unnecessary columns
df = df.drop(columns=['Timestamp', 'Record number'])

# Display info of the cleaned dataframe
print("\nInfo of the cleaned dataframe:")
df.info()

# Save the preprocessed data
df.to_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv", index=False)

print("\nPreprocessing complete. Cleaned data saved to brisbane-water-quality-preprocessed.csv")
