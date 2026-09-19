from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

file = PROJECT_ROOT / "data" / "raw" / "BLAT_ECOLX_Jacquier_2013.csv"

df = pd.read_csv(file)

# Extract mutation information
def parse_mutation(mutation):
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)

    if match:
        original = match.group(1)
        position = int(match.group(2))
        new = match.group(3)

        return pd.Series([original, position, new])

    return pd.Series([None, None, None])


df[["original_aa", "position", "new_aa"]] = df["mutant"].apply(parse_mutation)

print("=" * 50)
print("MUTATION FEATURES")
print("=" * 50)

print("\nFirst 20 mutations:")
print(
    df[
        ["mutant", "original_aa", "position", "new_aa", "DMS_score"]
    ].head(20).to_string(index=False)
)

print("\nUnique original amino acids:")
print(sorted(df["original_aa"].unique()))

print("\nUnique new amino acids:")
print(sorted(df["new_aa"].unique()))

print("\nPosition range:")
print(df["position"].min(), "to", df["position"].max())