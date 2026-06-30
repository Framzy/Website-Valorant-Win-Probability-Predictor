"""
Utility functions for Valorant Predictor V2 Training
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)

from xgboost import XGBRegressor

from training_v2.config import (
    AGENT_ROLE_MAP,
    CV_FOLDS,
    DATASET_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    ROLE_ORDER,
    TRAIN_TEST_SPLIT,
    XGB_PARAMS,
)

# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==========================================================
# LOGGING
# ==========================================================

def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warning(message: str) -> None:
    print(f"[WARNING] {message}")


# ==========================================================
# DATASET
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """
    Load raw dataset.
    """

    log_section("LOAD DATASET")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    df["Agent"] = (
        df["Agent"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    log_info(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# AGGREGATION
# ==========================================================

def aggregate_matches(
    df: pd.DataFrame,
) -> pd.DataFrame:

    log_section("AGGREGATE MATCHES")

    before = len(df)

    group_columns = [

        "Tournament",

        "Stage",

        "Match Type",

        "Map",

        "Team",

    ]
    
    df = df[
    (df["Stage"] == "All Stages")
    &
    (df["Match Type"] == "All Match Types")
    ].copy()

    after = len(df)
    log_info(f"Raw Rows        : {before}")
    log_info(f"All Stages Rows : {after}")
    
    grouped = (

        df.groupby(group_columns)

        .agg(

            {

                "Agent": list,

                "Total Wins By Map": "max",

                "Total Loss By Map": "max",

                "Total Maps Played": "max",

                "Year": "first",

            }

        )

        .reset_index()

    )

    return grouped


# ==========================================================
# VALIDATION
# ==========================================================

def validate_dataset(
    grouped: pd.DataFrame,
) -> None:

    log_section("VALIDATE DATASET")

    invalid_size = grouped[
        grouped["Agent"].apply(
            lambda agents: len(agents) != 5
        )
    ]
    
    invalid_size["Agent Count"] = (
    invalid_size["Agent"].apply(len)
)

    print(

        invalid_size[
            [
                "Team",
                "Map",
                "Tournament",
                "Agent Count",
            ]
        ]
        .head(30)

    )

    if not invalid_size.empty:

        log_warning(
            f"Found {len(invalid_size)} invalid compositions."
        )

        print(
            invalid_size[
                [
                    "Tournament",
                    "Stage",
                    "Map",
                    "Team",
                    "Agent"
                ]
            ].head(20)
        )

    duplicated = grouped[
        grouped["Agent"].apply(
            lambda x: len(set(x)) != 5
        )
    ]

    if not duplicated.empty:

        log_warning(

            f"{len(duplicated)} "

            "composition(s) contain duplicated agents."

        )

    log_info("Dataset validation passed.")


# ==========================================================
# TARGET
# ==========================================================

def calculate_target(
    grouped: pd.DataFrame,
) -> pd.DataFrame:

    log_section("CALCULATE WINRATE")

    denominator = (
        grouped["Total Wins By Map"]
        + grouped["Total Loss By Map"]
    )

    grouped["Winrate"] = np.where(
        denominator == 0,
        0.0,
        grouped["Total Wins By Map"] / denominator
    )

    log_info(
        f"Mean Winrate : "
        f"{grouped['Winrate'].mean():.4f}"
    )

    return grouped


# ==========================================================
# ROLE VECTOR
# ==========================================================

def get_role_vector(
    agents: list[str],
) -> list[int]:

    role_count = {

        role: 0

        for role in ROLE_ORDER

    }

    for agent in agents:

        role = AGENT_ROLE_MAP.get(agent)

        if role is not None:

            role_count[role] += 1

    return [

        role_count[role]

        for role in ROLE_ORDER

    ]
    
    
# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def encode_features(
    grouped: pd.DataFrame,
):
    """
    Build feature matrix and target.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    artifacts : dict
    """

    log_section("FEATURE ENGINEERING")

    # ------------------------------------------------------
    # Team Encoder
    # ------------------------------------------------------

    team_ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    team_encoded = team_ohe.fit_transform(
        grouped[["Team"]]
    )

    # ------------------------------------------------------
    # Map Encoder
    # ------------------------------------------------------

    map_ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    map_encoded = map_ohe.fit_transform(
        grouped[["Map"]]
    )

    # ------------------------------------------------------
    # Year Encoder
    # ------------------------------------------------------

    year_ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    year_encoded = year_ohe.fit_transform(
        grouped[["Year"]]
    )

    # ------------------------------------------------------
    # Agent Encoder
    # ------------------------------------------------------

    mlb = MultiLabelBinarizer()

    agent_matrix = mlb.fit_transform(
        grouped["Agent"]
    )

    # ------------------------------------------------------
    # Role Vector
    # ------------------------------------------------------

    role_matrix = np.vstack(

        grouped["Agent"].apply(
            get_role_vector
        )

    )

    role_scaler = StandardScaler()

    role_scaled = role_scaler.fit_transform(
        role_matrix
    )

    # ------------------------------------------------------
    # Feature Matrix
    # ------------------------------------------------------

    X = np.hstack([

        year_encoded,

        team_encoded,

        map_encoded,

        agent_matrix,

        role_scaled,

    ])

    y = grouped["Winrate"].astype(np.float32).to_numpy()

    # ------------------------------------------------------
    # Feature Names
    # ------------------------------------------------------

    feature_columns = []

    feature_columns.extend(
        year_ohe.get_feature_names_out(["Year"])
    )

    feature_columns.extend(
        team_ohe.get_feature_names_out(["Team"])
    )

    feature_columns.extend(
        map_ohe.get_feature_names_out(["Map"])
    )

    feature_columns.extend(
        [
            f"Agent_{agent}"
            for agent in mlb.classes_
        ]
    )

    feature_columns.extend(

        [
            "Role_Duelist",
            "Role_Initiator",
            "Role_Controller",
            "Role_Sentinel",
        ]

    )

    feature_columns = list(feature_columns)
    
    if len(feature_columns) != X.shape[1]:
        raise ValueError(
            "Feature column count does not match feature matrix."
        )

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    log_info(f"Samples  : {X.shape[0]}")

    log_info(f"Features : {X.shape[1]}")

    ratio = round(
        X.shape[0] / X.shape[1],
        2
    )

    log_info(f"Sample Ratio : {ratio}")

    # ------------------------------------------------------
    # Agent Coverage
    # ------------------------------------------------------

    log_section("AGENT COVERAGE")

    counter = {}

    for composition in grouped["Agent"]:

        for agent in composition:

            counter[agent] = (
                counter.get(agent, 0) + 1
            )

    coverage = (

        pd.Series(counter)

        .sort_values()

    )

    print(coverage)

    # ------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------

    artifacts = {

        "team_ohe": team_ohe,

        "map_ohe": map_ohe,

        "year_ohe": year_ohe,

        "mlb": mlb,

        "role_scaler": role_scaler,

        "agent_role_map": AGENT_ROLE_MAP,

        "feature_columns": feature_columns,
        
        "agent_classes": list(mlb.classes_)

    }

    return (

        X,

        y,

        artifacts,

    )
    
# ==========================================================
# MODEL TRAINING
# ==========================================================

def train_model(
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Train XGBoost model.
    """

    log_section("MODEL TRAINING")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )

    model = XGBRegressor(
        **XGB_PARAMS
    )

    log_info("Training model...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    log_info(
        f"MAE : {mae:.4f}"
    )

    metrics = {

        "mae": float(mae),

        "train_size": len(X_train),

        "test_size": len(X_test),

    }

    return (

        model,

        metrics,

    )


# ==========================================================
# CROSS VALIDATION
# ==========================================================

def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
):
    model = XGBRegressor(
    **XGB_PARAMS
    )

    log_section(
        "CROSS VALIDATION"
    )

    cv = KFold(

        n_splits=CV_FOLDS,

        shuffle=True,

        random_state=RANDOM_STATE,

    )

    scores = cross_val_score(

        model,

        X,

        y,

        cv=cv,

        scoring="neg_mean_absolute_error",

    )

    mae_scores = -scores

    print()

    print("MAE per Fold")

    print(mae_scores)

    mean_mae = float(
        mae_scores.mean()
    )

    std_mae = float(
        mae_scores.std()
    )

    print()

    print(
        f"Mean MAE : {mean_mae:.4f}"
    )

    print(
        f"Std MAE : {std_mae:.4f}"
    )

    return {

        "cv_scores": mae_scores.tolist(),

        "cv_mean": mean_mae,

        "cv_std": std_mae,

    }


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def print_feature_importance(
    model,
    feature_columns,
    top_n=20,
):

    log_section(
        "FEATURE IMPORTANCE"
    )

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_,

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False,

    )
    
    importance.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
    )

    print(
        importance.head(top_n)
    )

    return importance

# ==========================================================
# SAVE ARTIFACTS
# ==========================================================

def save_artifacts(
    model,
    artifacts: dict,
) -> None:
    """
    Save model and preprocessing artifacts.
    """

    log_section("SAVE ARTIFACTS")

    files = {

        "xgb_model.pkl": model,

        "team_ohe.pkl":
            artifacts["team_ohe"],

        "map_ohe.pkl":
            artifacts["map_ohe"],

        "year_ohe.pkl":
            artifacts["year_ohe"],

        "mlb.pkl":
            artifacts["mlb"],

        "role_scaler.pkl":
            artifacts["role_scaler"],

        "agent_role_map.pkl":
            artifacts["agent_role_map"],

        "feature_columns.pkl":
            artifacts["feature_columns"],

    }

    for filename, obj in files.items():

        path = MODEL_DIR / filename

        joblib.dump(
            obj,
            path,
        )

        log_info(
            f"Saved : {filename}"
        )


# ==========================================================
# MODEL METADATA
# ==========================================================

def save_model_metadata(
    metrics: dict,
    cv_metrics: dict,
    X: np.ndarray,
) -> None:

    log_section(
        "SAVE METADATA"
    )

    metadata = {

        "model": "XGBRegressor",

        "dataset_size":
            int(X.shape[0]),

        "feature_count":
            int(X.shape[1]),

        "mae":
            round(
                metrics["mae"],
                4,
            ),

        "cv_mean":
            round(
                cv_metrics["cv_mean"],
                4,
            ),

        "cv_std":
            round(
                cv_metrics["cv_std"],
                4,
            ),

        "created_at":
            datetime.now().isoformat(),

    }

    with open(

        MODEL_DIR / "model_metadata.json",

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4,

        )

    log_info(
        "Saved : model_metadata.json"
    )


# ==========================================================
# TRAINING SUMMARY
# ==========================================================

def save_training_summary(
    grouped: pd.DataFrame,
) -> None:

    log_section(
        "SAVE SUMMARY"
    )

    summary = {

        "total_compositions":

            int(len(grouped)),

        "total_teams":

            int(grouped["Team"].nunique()),

        "total_maps":

            int(grouped["Map"].nunique()),

        "year_distribution":

            grouped["Year"]
            .value_counts()
            .sort_index()
            .to_dict(),

    }

    with open(

        MODEL_DIR / "training_summary.json",

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            summary,

            f,

            indent=4,

        )

    log_info(
        "Saved : training_summary.json"
    )