from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


# ============================================================
# LOCKED CANDIDATE VALIDATION
# 5 POSITION-AWARE FOLDS × 3 REPEATS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_features.csv"

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "locked_candidate_cv_results.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "locked_candidate_cv_summary.csv"
)


# ============================================================
# LOCKED CANDIDATE — DO NOT CHANGE
# ============================================================

N_FOLDS = 5
N_REPEATS = 3

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.04,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def transform_target(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_transform_target(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.expm1(np.abs(y))


def create_sample_weights(y):
    y = np.asarray(y, dtype=float)

    weights = np.ones(len(y), dtype=float)
    weights[y <= -4] = 2.0

    return weights


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    extreme_mask = actual <= -4
    positive_mask = actual > 0

    metrics = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "bias": np.mean(errors),
        "median_absolute_error": np.median(np.abs(errors)),
        "max_absolute_error": np.max(np.abs(errors)),
        "extreme_mae": (
            mean_absolute_error(
                actual[extreme_mask],
                predicted[extreme_mask],
            )
            if extreme_mask.any()
            else np.nan
        ),
        "positive_mae": (
            mean_absolute_error(
                actual[positive_mask],
                predicted[positive_mask],
            )
            if positive_mask.any()
            else np.nan
        ),
        "n_samples": len(actual),
        "n_extreme": int(extreme_mask.sum()),
        "n_positive": int(positive_mask.sum()),
    }

    return metrics


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ENZYOPT-AI — LOCKED CANDIDATE VALIDATION")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

y = df["DMS_score"].astype(float)
groups = df["position"]

feature_columns = [
    c for c in df.columns
    if c not in ["DMS_score", "position"]
]

X = df[feature_columns].copy()

print(f"\nDataset: {df.shape}")
print(f"Samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Positions: {groups.nunique()}")


# ============================================================
# ORIGINAL LOCKED 80/20 HOLDOUT
# ============================================================

holdout_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, holdout_idx = next(
    holdout_splitter.split(
        X,
        y,
        groups=groups,
    )
)

train_positions = set(groups.iloc[train_idx])
holdout_positions = set(groups.iloc[holdout_idx])

assert not (train_positions & holdout_positions)

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
groups_train = groups.iloc[train_idx].reset_index(drop=True)

X_holdout = X.iloc[holdout_idx]
y_holdout = y.iloc[holdout_idx]

print("\nOriginal locked holdout:")
print(f"  Training samples:  {len(train_idx)}")
print(f"  Holdout samples:   {len(holdout_idx)}")
print(f"  Training positions:{len(train_positions)}")
print(f"  Holdout positions: {len(holdout_positions)}")
print("  Position overlap:  0")


# ============================================================
# REPEATED POSITION-AWARE CV
# ============================================================

all_results = []
all_predictions = []

for repeat in range(1, N_REPEATS + 1):

    print("\n" + "=" * 70)
    print(f"REPEAT {repeat}/{N_REPEATS}")
    print("=" * 70)

    # sklearn versions supporting shuffle=True give a
    # genuinely different group partition for each repeat.
    cv = GroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=1000 + repeat,
    )

    for fold, (cv_train_idx, cv_valid_idx) in enumerate(
        cv.split(
            X_train,
            y_train,
            groups=groups_train,
        ),
        start=1,
    ):

        cv_train_positions = set(
            groups_train.iloc[cv_train_idx]
        )

        cv_valid_positions = set(
            groups_train.iloc[cv_valid_idx]
        )

        overlap = cv_train_positions & cv_valid_positions

        if overlap:
            raise RuntimeError(
                f"POSITION LEAKAGE in repeat {repeat}, "
                f"fold {fold}: {overlap}"
            )

        X_cv_train = X_train.iloc[cv_train_idx]
        X_cv_valid = X_train.iloc[cv_valid_idx]

        y_cv_train = y_train.iloc[cv_train_idx]
        y_cv_valid = y_train.iloc[cv_valid_idx]

        # Exact locked target transformation.
        transformed_y = transform_target(y_cv_train)

        # Exact locked weighting.
        sample_weights = create_sample_weights(y_cv_train)

        # Exact locked candidate model.
        model = HistGradientBoostingRegressor(
            **MODEL_PARAMS
        )

        model.fit(
            X_cv_train,
            transformed_y,
            sample_weight=sample_weights,
        )

        transformed_predictions = model.predict(X_cv_valid)

        predictions = inverse_transform_target(
            transformed_predictions
        )

        metrics = calculate_metrics(
            y_cv_valid,
            predictions,
        )

        metrics.update({
            "repeat": repeat,
            "fold": fold,
            "train_samples": len(cv_train_idx),
            "valid_samples": len(cv_valid_idx),
            "train_positions": len(cv_train_positions),
            "valid_positions": len(cv_valid_positions),
        })

        all_results.append(metrics)

        for idx, actual, predicted in zip(
            cv_valid_idx,
            y_cv_valid.to_numpy(),
            predictions,
        ):
            error = predicted - actual

            all_predictions.append({
                "repeat": repeat,
                "fold": fold,
                "position": groups_train.iloc[idx],
                "actual": actual,
                "predicted": predicted,
                "error": error,
                "absolute_error": abs(error),
            })

        print(
            f"Repeat {repeat} | Fold {fold} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"R2={metrics['R2']:.4f} | "
            f"Bias={metrics['bias']:.4f} | "
            f"Extreme MAE={metrics['extreme_mae']:.4f}"
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)
predictions_df = pd.DataFrame(all_predictions)

results_df.to_csv(RESULTS_FILE, index=False)

print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

metric_columns = [
    "MAE",
    "RMSE",
    "R2",
    "bias",
    "extreme_mae",
    "positive_mae",
]

summary_rows = []

for metric in metric_columns:

    values = results_df[metric].dropna()

    summary_rows.append({
        "metric": metric,
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "min": values.min(),
        "max": values.max(),
        "median": values.median(),
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(SUMMARY_FILE, index=False)

print("\n5-fold × 3-repeat validation:")
print(summary_df.to_string(index=False))

print("\nRepeat-level means:")

repeat_summary = (
    results_df
    .groupby("repeat")[metric_columns]
    .mean()
)

print(repeat_summary.to_string())

print("\nOverall fold mean ± SD:")

for metric in metric_columns:
    mean = results_df[metric].mean()
    std = results_df[metric].std(ddof=1)

    print(
        f"{metric:20s}: "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# FIXED ORIGINAL HOLDOUT
# ============================================================

print("\n" + "=" * 70)
print("FIXED ORIGINAL HOLDOUT")
print("=" * 70)

# Train the candidate exactly as originally trained.
holdout_model = HistGradientBoostingRegressor(
    **MODEL_PARAMS
)

holdout_model.fit(
    X_train,
    transform_target(y_train),
    sample_weight=create_sample_weights(y_train),
)

holdout_predictions = inverse_transform_target(
    holdout_model.predict(X_holdout)
)

holdout_metrics = calculate_metrics(
    y_holdout,
    holdout_predictions,
)

for key, value in holdout_metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:24s}: {value:.4f}")
    else:
        print(f"{key:24s}: {value}")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CV vs ORIGINAL HOLDOUT")
print("=" * 70)

cv_means = results_df[metric_columns].mean()

for metric in metric_columns:

    cv_value = cv_means[metric]
    holdout_value = holdout_metrics[metric]

    difference = holdout_value - cv_value

    print(
        f"{metric:20s} | "
        f"CV mean={cv_value:.4f} | "
        f"Holdout={holdout_value:.4f} | "
        f"Holdout-CV={difference:+.4f}"
    )

print("\nSaved:")
print(RESULTS_FILE)
print(SUMMARY_FILE)

print("\nValidation complete.")