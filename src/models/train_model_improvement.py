from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# ENZYOPT-AI — MODEL IMPROVEMENT EXPERIMENT
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — MODEL IMPROVEMENT EXPERIMENT")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_improvement_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_improvement_predictions.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

if "DMS_score" not in df.columns:
    raise ValueError(
        "DMS_score column not found!"
    )

if "position" not in df.columns:
    raise ValueError(
        "position column not found!"
    )


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\nSamples:", len(df))
print("Features:", df.shape[1] - 1)
print("Unique positions:", df["position"].nunique())


# ============================================================
# 4. TARGET / FEATURES
# ============================================================

y = df["DMS_score"]

# Position is used ONLY for grouping.
# It is deliberately excluded from predictive features.
groups = df["position"]

X = df.drop(
    columns=["DMS_score", "position"]
)

print("\nPredictive feature matrix:")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

if X.isnull().sum().sum() != 0:
    raise ValueError(
        "Missing values detected in feature matrix!"
    )


# ============================================================
# 5. POSITION-AWARE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("POSITION-AWARE SPLIT")
print("=" * 60)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

train_positions = set(
    groups.iloc[train_idx]
)

test_positions = set(
    groups.iloc[test_idx]
)

overlap = (
    train_positions
    & test_positions
)

print("\nTraining samples:", len(train_idx))
print("Testing samples :", len(test_idx))
print("Training positions:", len(train_positions))
print("Testing positions :", len(test_positions))
print("Overlapping positions:", len(overlap))

if overlap:
    raise ValueError(
        "Position leakage detected!"
    )

print("No position overlap confirmed.")


# ============================================================
# 6. TRAIN / TEST DATA
# ============================================================

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ============================================================
# 7. EXTREME SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nTraining target:")
print(y_train.describe())

print("\nTesting target:")
print(y_test.describe())

print(
    "\nVery low scores (<= -4):",
    int((y_test <= -4).sum())
)

print(
    "Low scores (-4 to -2):",
    int(((y_test > -4) & (y_test <= -2)).sum())
)

print(
    "Mild negative (-2 to 0):",
    int(((y_test > -2) & (y_test < 0)).sum())
)

print(
    "Zero:",
    int((y_test == 0).sum())
)

print(
    "Positive (> 0):",
    int((y_test > 0).sum())
)


# ============================================================
# 8. SAMPLE WEIGHTS
# ============================================================

# We do NOT remove extreme observations.
#
# Instead, this experiment gives additional training weight
# to extreme target values.
#
# The weight increases gradually with distance from zero.

sample_weights = 1.0 + np.clip(
    np.abs(y_train.to_numpy()),
    0,
    5,
) / 2.0

print("\nSample-weight range:")
print(
    "Minimum:",
    sample_weights.min()
)

print(
    "Maximum:",
    sample_weights.max()
)


# ============================================================
# 9. MODEL DEFINITIONS
# ============================================================

models = {

    "Random Forest": (
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        False,
    ),

    "Random Forest Weighted": (
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        True,
    ),

    "Extra Trees": (
        ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        False,
    ),

    "Extra Trees Weighted": (
        ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        True,
    ),

    "HistGradientBoosting": (
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
        False,
    ),
}


# ============================================================
# 10. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    errors = (
        predictions - y_true
    )

    absolute_errors = np.abs(
        errors
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = mean_squared_error(
        y_true,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_true,
        predictions,
    )

    bias = np.mean(
        errors
    )

    median_absolute_error = np.median(
        absolute_errors
    )

    max_absolute_error = np.max(
        absolute_errors
    )

    # --------------------------------------------------------
    # Extreme-score metrics
    # --------------------------------------------------------

    extreme_mask = (
        y_true <= -4
    )

    if extreme_mask.sum() > 0:

        extreme_mae = mean_absolute_error(
            y_true[extreme_mask],
            predictions[extreme_mask],
        )

        extreme_rmse = (
            mean_squared_error(
                y_true[extreme_mask],
                predictions[extreme_mask],
            )
            ** 0.5
        )

        extreme_bias = np.mean(
            errors[extreme_mask]
        )

    else:

        extreme_mae = np.nan
        extreme_rmse = np.nan
        extreme_bias = np.nan

    # --------------------------------------------------------
    # Positive-score metrics
    # --------------------------------------------------------

    positive_mask = (
        y_true > 0
    )

    if positive_mask.sum() > 0:

        positive_mae = mean_absolute_error(
            y_true[positive_mask],
            predictions[positive_mask],
        )

    else:

        positive_mae = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "bias": bias,
        "median_absolute_error": median_absolute_error,
        "max_absolute_error": max_absolute_error,
        "extreme_mae": extreme_mae,
        "extreme_rmse": extreme_rmse,
        "extreme_bias": extreme_bias,
        "positive_mae": positive_mae,
        "n_extreme": int(
            extreme_mask.sum()
        ),
        "n_positive": int(
            positive_mask.sum()
        ),
    }


# ============================================================
# 11. TRAIN MODELS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL EXPERIMENTS")
print("=" * 60)

results = []
prediction_rows = []

for model_name, (
    model,
    use_weights,
) in models.items():

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(
        "Weighted training:",
        use_weights,
    )

    if use_weights:

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

    else:

        model.fit(
            X_train,
            y_train,
        )

    print("Training complete.")

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print(
        f"Bias : {metrics['bias']:.4f}"
    )

    print(
        f"Extreme MAE (<= -4): "
        f"{metrics['extreme_mae']:.4f}"
    )

    print(
        f"Extreme RMSE (<= -4): "
        f"{metrics['extreme_rmse']:.4f}"
    )

    print(
        f"Positive-score MAE: "
        f"{metrics['positive_mae']:.4f}"
    )

    results.append(
        {
            "model": model_name,
            "n_features": X.shape[1],
            **metrics,
        }
    )

    # --------------------------------------------------------
    # Save individual predictions
    # --------------------------------------------------------

    for row_idx, prediction in zip(
        test_idx,
        predictions,
    ):

        actual = df.iloc[row_idx][
            "DMS_score"
        ]

        prediction_rows.append(
            {
                "model": model_name,
                "position": df.iloc[row_idx][
                    "position"
                ],
                "actual": actual,
                "predicted": prediction,
                "error": prediction - actual,
                "absolute_error": abs(
                    prediction - actual
                ),
            }
        )


# ============================================================
# 12. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True,
)

print("\n" + "=" * 60)
print("MODEL IMPROVEMENT RESULTS")
print("=" * 60)

print(
    "\n"
    + results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 13. COMPARE AGAINST PREVIOUS MODEL
# ============================================================

previous_mae = 1.2685
previous_rmse = 1.6095
previous_r2 = 0.1684

print("\n" + "=" * 60)
print("COMPARISON WITH PREVIOUS ALL-FEATURE MODEL")
print("=" * 60)

print(
    f"\nPrevious MAE : {previous_mae:.4f}"
)

print(
    f"Previous RMSE: {previous_rmse:.4f}"
)

print(
    f"Previous R²  : {previous_r2:.4f}"
)

comparison_rows = []

for _, row in results_df.iterrows():

    comparison_rows.append(
        {
            "model": row["model"],
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"],
            "MAE_change": (
                row["MAE"]
                - previous_mae
            ),
            "RMSE_change": (
                row["RMSE"]
                - previous_rmse
            ),
            "R2_change": (
                row["R2"]
                - previous_r2
            ),
            "extreme_MAE": row[
                "extreme_mae"
            ],
            "extreme_RMSE": row[
                "extreme_rmse"
            ],
        }
    )

comparison_df = pd.DataFrame(
    comparison_rows
)

print(
    "\n"
    + comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

print("\nResults saved to:")
print(RESULTS_FILE)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

print("\nPredictions saved to:")
print(PREDICTIONS_FILE)


# ============================================================
# 16. EXTREME ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXTREME-SCORE ERROR ANALYSIS")
print("=" * 60)

extreme_predictions = (
    prediction_df[
        prediction_df["actual"] <= -4
    ]
    .sort_values(
        "absolute_error",
        ascending=False,
    )
)

if len(extreme_predictions) > 0:

    print(
        "\nWorst predictions for "
        "DMS_score <= -4:"
    )

    print(
        extreme_predictions.head(
            20
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

else:

    print(
        "\nNo test observations with "
        "DMS_score <= -4."
    )


# ============================================================
# 17. OVERALL WORST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("WORST INDIVIDUAL PREDICTIONS")
print("=" * 60)

worst_predictions = (
    prediction_df
    .sort_values(
        "absolute_error",
        ascending=False,
    )
    .head(20)
)

print(
    "\n"
    + worst_predictions.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("MODEL IMPROVEMENT COMPLETE")
print("=" * 60)

print(
    "\nBest model by MAE in this experiment:"
)

print(
    best_model["model"]
)

print(
    f"\nMAE : {best_model['MAE']:.4f}"
)

print(
    f"RMSE: {best_model['RMSE']:.4f}"
)

print(
    f"R²  : {best_model['R2']:.4f}"
)

print(
    f"Extreme MAE (<= -4): "
    f"{best_model['extreme_mae']:.4f}"
)

print(
    "\nAll models were evaluated on the "
    "same position-aware test split."
)

print(
    "Extreme observations were retained; "
    "weighted models only changed their "
    "training importance."
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)