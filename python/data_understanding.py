
import pandas as pd

# Load the dataset
df = pd.read_csv("job_acceptance_prediction.csv")

# Dataset shape
print("Dataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# First 5 records
print("\nFirst 5 Records:")
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())
