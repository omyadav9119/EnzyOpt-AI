from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# ENZYOPT-AI — LOCKED CANDIDATE DIAGNOSTIC
# ============================================================
# Purpose:
#   Diagnose WHY repeated position-aware CV (~1.47 MAE, ~0 R2)
#   is much worse than the original fixed holdout (1.0914 MAE,
#   0.3040 R2).
#
# IMPORTANT:
#   This script does NOT tune or change the candidate model.
#   It uses the exact locked candidate configuration.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_features.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

PRED_FILE = OUT_DIR / "locked_candidate_diagnostic_predictions.csv"
POS_FILE = OUT_DIR / "locked_candidate_position_difficulty.csv"
DIST_FILE = OUT_DIR / "locked_candidate_position_distribution.csv"
PROXY_FILE = OUT_DIR / "locked_candidate_position_proxy_features.csv"
FOLD_FILE = OUT_DIR / "locked_candidate_fold_summary.csv"


# ============================================================
# LOCKED MODEL — DO NOT CHANGE
# ============================================================

MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=15,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=42,
)


def transform_target(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_transform_target(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.expm1(np.abs(y))


def sample_weights(y):
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y), dtype=float)
    w[y <= -4] = 2.0
    return w


def build_model():
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


def metric_row(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual

    extreme = actual <= -4
    positive = actual > 0

    return {
        "n": len(actual),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred),
        "bias": np.mean(err),
        "extreme_mae": (
            mean_absolute_error(actual[extreme], pred[extreme])
            if extreme.any() else np.nan
        ),
        "positive_mae": (
            mean_absolute_error(actual[positive], pred[positive])
            if positive.any() else np.nan
        ),
        "n_extreme": int(extreme.sum()),
        "n_positive": int(positive.sum()),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ENZYOPT-AI — LOCKED CANDIDATE DIAGNOSTIC")
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
print(f"Features: {len(feature_columns)}")
print(f"Positions: {groups.nunique()}")


# ============================================================
# REPRODUCE ORIGINAL LOCKED HOLDOUT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, holdout_idx = next(
    splitter.split(X, y, groups=groups)
)

train_idx = np.asarray(train_idx)
holdout_idx = np.asarray(holdout_idx)

train_positions = set(groups.iloc[train_idx])
holdout_positions = set(groups.iloc[holdout_idx])

assert not (train_positions & holdout_positions)

print("\nOriginal split reproduced:")
print(f"  Train samples:   {len(train_idx)}")
print(f"  Holdout samples: {len(holdout_idx)}")
print(f"  Train positions: {len(train_positions)}")
print(f"  Holdout positions: {len(holdout_positions)}")


# ============================================================
# 1. FIXED HOLDOUT PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("1. FIXED HOLDOUT PREDICTIONS")
print("=" * 70)

model = build_model()

model.fit(
    X.iloc[train_idx],
    transform_target(y.iloc[train_idx]),
    sample_weight=sample_weights(y.iloc[train_idx]),
)

holdout_pred = inverse_transform_target(
    model.predict(X.iloc[holdout_idx])
)

holdout_metrics = metric_row(
    y.iloc[holdout_idx],
    holdout_pred,
)

print(
    f"Holdout MAE={holdout_metrics['MAE']:.4f} | "
    f"RMSE={holdout_metrics['RMSE']:.4f} | "
    f"R2={holdout_metrics['R2']:.4f}"
)


# ============================================================
# 2. REPEATED POSITION-AWARE CV PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("2. REPEATED POSITION-AWARE CV")
print("=" * 70)

cv_df = df.iloc[train_idx].copy()
X_cv = X.iloc[train_idx]
y_cv = y.iloc[train_idx]
groups_cv = groups.iloc[train_idx]

prediction_rows = []
fold_rows = []

for repeat in range(1, 4):
    splitter = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=1000 + repeat,
    )

    for fold, (inner_train, inner_valid) in enumerate(
        splitter.split(X_cv, y_cv, groups=groups_cv),
        start=1,
    ):
        inner_train = np.asarray(inner_train)
        inner_valid = np.asarray(inner_valid)

        train_groups = set(groups_cv.iloc[inner_train])
        valid_groups = set(groups_cv.iloc[inner_valid])

        assert not (train_groups & valid_groups)

        m = build_model()

        m.fit(
            X_cv.iloc[inner_train],
            transform_target(y_cv.iloc[inner_train]),
            sample_weight=sample_weights(y_cv.iloc[inner_train]),
        )

        pred = inverse_transform_target(
            m.predict(X_cv.iloc[inner_valid])
        )

        metrics = metric_row(y_cv.iloc[inner_valid], pred)

        fold_rows.append({
            "repeat": repeat,
            "fold": fold,
            **metrics,
        })

        valid_global_indices = train_idx[inner_valid]

        for global_i, pred_i in zip(valid_global_indices, pred):
            actual_i = float(df.iloc[global_i]["DMS_score"])
            position_i = df.iloc[global_i]["position"]

            prediction_rows.append({
                "repeat": repeat,
                "fold": fold,
                "global_index": int(global_i),
                "position": position_i,
                "actual": actual_i,
                "predicted": float(pred_i),
                "error": float(pred_i - actual_i),
                "absolute_error": float(abs(pred_i - actual_i)),
            })

        print(
            f"Repeat {repeat} Fold {fold}: "
            f"MAE={metrics['MAE']:.4f} | "
            f"R2={metrics['R2']:.4f} | "
            f"n={metrics['n']}"
        )

pred_df = pd.DataFrame(prediction_rows)
fold_df = pd.DataFrame(fold_rows)

pred_df.to_csv(PRED_FILE, index=False)
fold_df.to_csv(FOLD_FILE, index=False)


# ============================================================
# 3. POSITION-LEVEL DIFFICULTY
# ============================================================

print("\n" + "=" * 70)
print("3. POSITION-LEVEL DIFFICULTY")
print("=" * 70)

# Every training position appears in validation exactly once per repeat.
position_rows = []

for position, g in pred_df.groupby("position"):
    actual = g["actual"].to_numpy()
    pred = g["predicted"].to_numpy()

    position_rows.append({
        "position": position,
        "n_predictions": len(g),
        "mean_actual": actual.mean(),
        "std_actual": actual.std(),
        "min_actual": actual.min(),
        "max_actual": actual.max(),
        "extreme_rate": np.mean(actual <= -4),
        "positive_rate": np.mean(actual > 0),
        "MAE": mean_absolute_error(actual, pred),
        "bias": np.mean(pred - actual),
        "mean_predicted": pred.mean(),
    })

position_df = pd.DataFrame(position_rows)

# Stability across the three repeats.
repeat_position = (
    pred_df
    .groupby(["position", "repeat"])
    .apply(
        lambda g: pd.Series({
            "repeat_mae": mean_absolute_error(
                g["actual"], g["predicted"]
            ),
            "repeat_bias": np.mean(
                g["predicted"] - g["actual"]
            ),
            "repeat_n": len(g),
        }),
        include_groups=False,
    )
    .reset_index()
)

repeat_summary = (
    repeat_position
    .groupby("position")
    .agg(
        repeat_mae_mean=("repeat_mae", "mean"),
        repeat_mae_std=("repeat_mae", "std"),
        repeat_bias_mean=("repeat_bias", "mean"),
    )
    .reset_index()
)

position_df = position_df.merge(
    repeat_summary,
    on="position",
    how="left",
)

position_df["difficulty_rank"] = (
    position_df["MAE"].rank(
        ascending=False,
        method="min",
    )
)

position_df = position_df.sort_values(
    ["MAE", "n_predictions"],
    ascending=[False, False],
)

position_df.to_csv(POS_FILE, index=False)

print("\nHardest 20 positions by CV MAE:")
print(
    position_df[
        [
            "position",
            "n_predictions",
            "mean_actual",
            "extreme_rate",
            "positive_rate",
            "MAE",
            "bias",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 4. HOLDOUT VS TRAINING POSITION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. HOLDOUT VS TRAINING POSITION DISTRIBUTION")
print("=" * 70)

all_position_stats = (
    df.groupby("position")["DMS_score"]
    .agg(
        n="size",
        mean="mean",
        std="std",
        min="min",
        max="max",
    )
    .reset_index()
)

all_position_stats["extreme_rate"] = (
    df.groupby("position")["DMS_score"]
    .apply(lambda s: np.mean(s <= -4))
    .values
)

all_position_stats["positive_rate"] = (
    df.groupby("position")["DMS_score"]
    .apply(lambda s: np.mean(s > 0))
    .values
)

all_position_stats["split"] = np.where(
    all_position_stats["position"].isin(holdout_positions),
    "fixed_holdout",
    "original_training",
)

dist_df = all_position_stats.copy()

# Add whether a position was observed as hard/easy by CV.
dist_df = dist_df.merge(
    position_df[
        [
            "position",
            "MAE",
            "repeat_mae_mean",
            "repeat_mae_std",
        ]
    ],
    on="position",
    how="left",
    suffixes=("_target", "_cv"),
)

dist_df.to_csv(DIST_FILE, index=False)

print("\nPosition-level target summary:")
print(
    dist_df.groupby("split")[
        ["n", "mean", "std", "extreme_rate", "positive_rate"]
    ].mean().to_string()
)

print("\nFixed holdout positions:")
print(
    dist_df[dist_df["split"] == "fixed_holdout"][
        [
            "position",
            "n",
            "mean",
            "std",
            "min",
            "max",
            "extreme_rate",
            "positive_rate",
        ]
    ]
    .sort_values("extreme_rate", ascending=False)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 5. POSITION-PROXY / FEATURE DEPENDENCE DIAGNOSTIC
# ============================================================
#
# We quantify how much each feature varies BETWEEN positions
# relative to its TOTAL variation.
#
# eta_squared = between-position SS / total SS
#
# High values mean the feature is strongly position-dependent.
# This does NOT prove leakage. It identifies features worth
# inspecting as potential position proxies.
# ============================================================

print("\n" + "=" * 70)
print("5. POSITION-PROXY FEATURE DIAGNOSTIC")
print("=" * 70)

proxy_rows = []

position_codes = pd.factorize(groups)[0]

for feature in feature_columns:
    values = X[feature].to_numpy(dtype=float)

    total_mean = np.mean(values)
    total_ss = np.sum((values - total_mean) ** 2)

    if total_ss == 0:
        continue

    tmp = pd.DataFrame({
        "group": position_codes,
        "value": values,
    })

    group_stats = tmp.groupby("group")["value"].agg(
        ["mean", "size"]
    )

    between_ss = np.sum(
        group_stats["size"]
        * (group_stats["mean"] - total_mean) ** 2
    )

    eta2 = between_ss / total_ss

    # Number of distinct values is useful for identifying
    # one-hot/categorical-like position encodings.
    n_unique = pd.Series(values).nunique()

    proxy_rows.append({
        "feature": feature,
        "eta_squared_position": eta2,
        "n_unique": n_unique,
        "mean": np.mean(values),
        "std": np.std(values),
    })

proxy_df = pd.DataFrame(proxy_rows).sort_values(
    "eta_squared_position",
    ascending=False,
)

proxy_df.to_csv(PROXY_FILE, index=False)

print("\nTop 30 most position-dependent features:")
print(
    proxy_df.head(30).to_string(index=False)
)


# ============================================================
# 6. SUMMARY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("6. DIAGNOSTIC SUMMARY")
print("=" * 70)

cv_mean = fold_df["MAE"].mean()
cv_std = fold_df["MAE"].std()
cv_r2 = fold_df["R2"].mean()

print(f"\nFixed holdout:")
print(f"  MAE = {holdout_metrics['MAE']:.4f}")
print(f"  R2  = {holdout_metrics['R2']:.4f}")

print("\nRepeated position-aware CV:")
print(f"  MAE = {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  R2  = {cv_r2:.4f}")

print("\nHoldout-vs-CV MAE gap:")
print(f"  {holdout_metrics['MAE'] - cv_mean:+.4f}")

print("\nHoldout-vs-CV R2 gap:")
print(f"  {holdout_metrics['R2'] - cv_r2:+.4f}")

print("\nPosition difficulty:")
print(
    f"  Median position MAE = "
    f"{position_df['MAE'].median():.4f}"
)
print(
    f"  Worst position MAE = "
    f"{position_df['MAE'].max():.4f}"
)
print(
    f"  Best position MAE = "
    f"{position_df['MAE'].min():.4f}"
)

print("\nTop position-proxy feature eta²:")
print(
    f"  {proxy_df.iloc[0]['feature']} = "
    f"{proxy_df.iloc[0]['eta_squared_position']:.4f}"
)

print("\nSaved:")
print(PRED_FILE)
print(POS_FILE)
print(DIST_FILE)
print(PROXY_FILE)
print(FOLD_FILE)

print("\nDiagnostic complete.")
