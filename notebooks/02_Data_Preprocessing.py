
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\drinking-water-quality-distribution-monitoring-data.csv", low_memory=False)

# 1. Clean Target Variables
# Define deterioration as any value not '<1' in Coliform or E.coli
def is_deteriorated(row):
    coliform = str(row['Coliform (Quanti-Tray) (MPN /100mL)']).strip()
    ecoli = str(row['E.coli(Quanti-Tray) (MPN/100mL)']).strip()
    if coliform != '<1' or ecoli != '<1':
        return 1
    return 0

df['Deterioration'] = df.apply(is_deteriorated, axis=1)

print("Value counts of Deterioration:")
print(df['Deterioration'].value_counts())

# 2. Clean Numerical Columns
# Convert Turbidity to numeric, coercing errors
df['Turbidity (NTU)'] = pd.to_numeric(df['Turbidity (NTU)'], errors='coerce')

# Handle negative chlorine values
df.loc[df['Residual Free Chlorine (mg/L)'] < 0, 'Residual Free Chlorine (mg/L)'] = np.nan

# 3. Handle Missing Values
# Drop Fluoride column
df = df.drop(columns=['Fluoride (mg/L)'])

# Impute missing values with the mean
df['Turbidity (NTU)'].fillna(df['Turbidity (NTU)'].mean(), inplace=True)
df['Residual Free Chlorine (mg/L)'].fillna(df['Residual Free Chlorine (mg/L)'].mean(), inplace=True)

# 4. Feature Engineering (Date/Time)
def clean_time(time_str):
    time_str = str(time_str).strip()
    if time_str == ':':
        return None
    if len(time_str) == 4 and time_str.isdigit():
        return time_str[:2] + ':' + time_str[2:]
    return time_str

df['Sample Time'] = df['Sample Time'].apply(clean_time)

df['Sample_DateTime'] = pd.to_datetime(df['Sample Date'] + ' ' + df['Sample Time'], errors='coerce')
df.dropna(subset=['Sample_DateTime'], inplace=True)

df['Sample_Month'] = df['Sample_DateTime'].dt.month
df['Sample_DayOfWeek'] = df['Sample_DateTime'].dt.dayofweek
df['Sample_Hour'] = df['Sample_DateTime'].dt.hour

# 5. Categorical Features
print("\nNumber of unique Sample Sites:", df['Sample Site'].nunique())
print("Number of unique Locations:", df['Location'].nunique())

# Drop original date/time and high cardinality columns for baseline model
df = df.drop(columns=['Sample Number', 'Sample Date', 'Sample Time', 'Sample Site', 'Location', 'Sample_DateTime', 'Coliform (Quanti-Tray) (MPN /100mL)', 'E.coli(Quanti-Tray) (MPN/100mL)'])

# Display info of the cleaned dataframe
print("\nInfo of the cleaned dataframe:")
df.info()

# Save the preprocessed data
df.to_csv(r"C:\Users\DELL\Desktop\predict\drinking-water-quality-preprocessed.csv", index=False)

print("\nPreprocessing complete. Cleaned data saved to drinking-water-quality-preprocessed.csv")
