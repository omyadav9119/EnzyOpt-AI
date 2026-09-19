from pathlib import Path

import pandas as pd

from amino_acid_properties import AMINO_ACID_PROPERTIES


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_context_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_features.csv"
)


# ============================================================
# 2. AMINO ACID GROUPS
# ============================================================

HYDROPHOBIC = set("AVILMFWY")
POLAR = set("STNQCY")
CHARGED = set("DEKR")
AROMATIC = set("FWY")
SMALL = set("AGSTC")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — CONTEXT FEATURE GENERATION")
print("=" * 60)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 4. CONTEXT RESIDUES
# ============================================================

context_columns = [
    "left_2",
    "left_1",
    "right_1",
    "right_2",
]


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def get_property(aa, property_name):
    return AMINO_ACID_PROPERTIES[aa][property_name]


def fraction_in_group(residues, group):
    valid = [aa for aa in residues if aa in AMINO_ACID_PROPERTIES]

    if not valid:
        return 0.0

    return sum(aa in group for aa in valid) / len(valid)


# ============================================================
# 6. CREATE CONTEXT FEATURES
# ============================================================

feature_rows = []

for _, row in df.iterrows():

    residues = [
        row["left_2"],
        row["left_1"],
        row["right_1"],
        row["right_2"],
    ]

    valid_residues = [
        aa for aa in residues
        if aa in AMINO_ACID_PROPERTIES
    ]

    # --------------------------------------------------------
    # Local composition
    # --------------------------------------------------------

    hydrophobic_fraction = fraction_in_group(
        valid_residues,
        HYDROPHOBIC
    )

    polar_fraction = fraction_in_group(
        valid_residues,
        POLAR
    )

    charged_fraction = fraction_in_group(
        valid_residues,
        CHARGED
    )

    aromatic_fraction = fraction_in_group(
        valid_residues,
        AROMATIC
    )

    small_fraction = fraction_in_group(
        valid_residues,
        SMALL
    )

    # --------------------------------------------------------
    # Local physicochemical averages
    # --------------------------------------------------------

    mean_hydrophobicity = sum(
        get_property(aa, "hydrophobicity")
        for aa in valid_residues
    ) / len(valid_residues)

    mean_molecular_weight = sum(
        get_property(aa, "molecular_weight")
        for aa in valid_residues
    ) / len(valid_residues)

    mean_polarity = sum(
        get_property(aa, "polarity")
        for aa in valid_residues
    ) / len(valid_residues)

    mean_charge = sum(
        get_property(aa, "charge")
        for aa in valid_residues
    ) / len(valid_residues)

    # --------------------------------------------------------
    # Mutation vs local environment
    # --------------------------------------------------------

    new_aa = row["new_aa"]

    new_hydrophobicity = get_property(
        new_aa,
        "hydrophobicity"
    )

    new_molecular_weight = get_property(
        new_aa,
        "molecular_weight"
    )

    new_polarity = get_property(
        new_aa,
        "polarity"
    )

    new_charge = get_property(
        new_aa,
        "charge"
    )

    feature_rows.append({
        "mutant": row["mutant"],
        "position": row["position"],
        "original_aa": row["original_aa"],
        "new_aa": row["new_aa"],

        "local_hydrophobic_fraction":
            hydrophobic_fraction,

        "local_polar_fraction":
            polar_fraction,

        "local_charged_fraction":
            charged_fraction,

        "local_aromatic_fraction":
            aromatic_fraction,

        "local_small_fraction":
            small_fraction,

        "local_mean_hydrophobicity":
            mean_hydrophobicity,

        "local_mean_molecular_weight":
            mean_molecular_weight,

        "local_mean_polarity":
            mean_polarity,

        "local_mean_charge":
            mean_charge,

        "new_vs_local_hydrophobicity":
            new_hydrophobicity - mean_hydrophobicity,

        "new_vs_local_molecular_weight":
            new_molecular_weight - mean_molecular_weight,

        "new_vs_local_polarity":
            new_polarity - mean_polarity,

        "new_vs_local_charge":
            new_charge - mean_charge,

        "DMS_score": row["DMS_score"],
    })


# ============================================================
# 7. CREATE DATAFRAME
# ============================================================

context_features = pd.DataFrame(feature_rows)

print("\nContext feature shape:")
print(context_features.shape)

print("\nColumns:")
print(context_features.columns.tolist())


# ============================================================
# 8. VALIDATION
# ============================================================

print("\nChecking missing values...")

missing = context_features.isnull().sum().sum()

print("Total missing values:", missing)

if missing != 0:
    raise ValueError("Missing values detected!")

print("\nFirst 20 rows:")
print(
    context_features.head(20).to_string(index=False)
)


# ============================================================
# 9. SAVE
# ============================================================

context_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nSaved context features to:")
print(OUTPUT_FILE)