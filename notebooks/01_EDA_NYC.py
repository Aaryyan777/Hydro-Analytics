
import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\drinking-water-quality-distribution-monitoring-data.csv")

# Print the head of the dataframe
print("Head of the dataframe:")
print(df.head())

# Print the info of the dataframe
print("\nInfo of the dataframe:")
df.info()

# Print the summary statistics
print("\nSummary statistics:")
print(df.describe())

# Print the value counts of the target variables
print("\nValue counts of Coliform:")
print(df['Coliform (Quanti-Tray) (MPN /100mL)'].value_counts())

print("\nValue counts of E.coli:")
print(df['E.coli(Quanti-Tray) (MPN/100mL)'].value_counts())

# Check for missing values
print("\nMissing values percentage:")
print(df.isnull().sum() / len(df) * 100)
