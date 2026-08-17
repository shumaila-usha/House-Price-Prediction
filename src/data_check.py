import pandas as pd

# Load the house price dataset
data = pd.read_csv("data/train.csv")

# Show basic information
print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())

print("\nTarget column:")
print("SalePrice")