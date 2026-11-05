import pandas as pd

# Load dataset
df = pd.read_csv("job_acceptance_prediction.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Separate numerical and categorical columns
categorical_cols = df.select_dtypes(include="str").columns

# Fill missing numerical values with median
numerical_cols = df.select_dtypes(include="number").columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Check missing values
print("Missing Values After Cleaning:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicates After Cleaning:")
print(df.duplicated().sum())

# Save cleaned dataset
df.to_csv("cleaned_job_acceptance.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Final Dataset Shape:", df.shape)