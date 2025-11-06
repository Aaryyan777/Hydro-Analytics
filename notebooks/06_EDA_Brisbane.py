
import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane_water_quality.csv")

# Print the head of the dataframe
print("Head of the dataframe:")
print(df.head())

# Print the info of the dataframe
print("\nInfo of the dataframe:")
df.info()

# Print the summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values percentage:")
print(df.isnull().sum() / len(df) * 100)
