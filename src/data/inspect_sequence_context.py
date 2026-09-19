from pathlib import Path
import pandas as pd


# ============================================================
# ENZYOPT-AI — SEQUENCE CONTEXT INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "encoded_sequence_context.csv"
)


print("=" * 60)
print("ENZYOPT-AI — SEQUENCE CONTEXT INSPECTION")
print("=" * 60)

print("\nLoading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 20 rows:")
print(df.head(20).to_string(index=False))

print("\nUnique values per column:")

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].nunique())

print("\nUnique values for categorical/context columns:")

for column in ["left_2", "left_1", "right_1", "right_2", "new_aa"]:
    if column in df.columns:
        print(f"\n{column}:")
        print(sorted(df[column].dropna().unique()))

print("\nDMS score summary:")
print(df["DMS_score"].describe())

print("\nPosition summary:")
print(df["position"].describe())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)