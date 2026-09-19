from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

file = PROJECT_ROOT / "data" / "raw" / "BLAT_ECOLX_Jacquier_2013.csv"

df = pd.read_csv(file)

print("=" * 50)
print("TEM-1 DATASET")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)