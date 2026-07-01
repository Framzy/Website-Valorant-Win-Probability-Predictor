"""
Feature Engineering V2
======================
"""

import pandas as pd
import numpy as np
import ast
from training_v2.config import (
    TEAM_DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN  
)
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def load_dataset() -> pd.DataFrame:
    """
    Load normalized training dataset.
    """

    print("=" * 60)
    print("LOAD TRAINING DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)
    df["Agent"] = df["Agent"].apply(ast.literal_eval)

    print(f"[INFO] Shape : {df.shape}")

    return df

def validate_dataset(
    df: pd.DataFrame
):
    """
    Validate normalized dataset.
    """

    print("\n" + "=" * 60)
    print("VALIDATE DATASET")
    print("=" * 60)

    invalid = df[
        df["Agent"].apply(len) != 5
    ]

    print(f"[INFO] Invalid Composition : {len(invalid)}")

    if len(invalid):

        raise ValueError(
            "Dataset still contains "
            "composition != 5 agents."
        )

    print("[INFO] Validation Passed")
    
def split_feature_target(
    df: pd.DataFrame,
):
    """
    Split feature and target.
    """

    print("\n" + "=" * 60)
    print("SPLIT FEATURE & TARGET")
    print("=" * 60)

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    print(f"[INFO] Feature Shape : {X.shape}")

    print(f"[INFO] Target Shape  : {y.shape}")

    return X, y

def build_encoders(
    X: pd.DataFrame,
):
    """
    Build feature encoders.
    """

    print("\n" + "=" * 60)
    print("BUILD ENCODERS")
    print("=" * 60)

    team_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    map_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    year_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    agent_encoder = MultiLabelBinarizer()

    team_encoder.fit(X[["Team"]])

    map_encoder.fit(X[["Map"]])

    year_encoder.fit(X[["Year"]])

    agent_encoder.fit(X["Agent"])

    print(f"[INFO] Team Category : {len(team_encoder.categories_[0])}")

    print(f"[INFO] Map Category  : {len(map_encoder.categories_[0])}")

    print(f"[INFO] Meta Category : {len(year_encoder.categories_[0])}")

    print(f"[INFO] Agent Count   : {len(agent_encoder.classes_)}")

    return {

        "team": team_encoder,

        "map": map_encoder,

        "year": year_encoder,

        "agent": agent_encoder,

    }    
    
def encode_features(
    X: pd.DataFrame,
    encoders: dict,
):
    """
    Encode all features into
    numerical feature matrix.
    """

    print("\n" + "=" * 60)
    print("ENCODE FEATURES")
    print("=" * 60)

    team_feature = encoders["team"].transform(
        X[["Team"]]
    )

    map_feature = encoders["map"].transform(
        X[["Map"]]
    )

    year_feature = encoders["year"].transform(
        X[["Year"]]
    )

    agent_feature = encoders["agent"].transform(
        X["Agent"]
    )

    X_encoded = np.concatenate(
        [
            team_feature,
            map_feature,
            year_feature,
            agent_feature,
        ],
        axis=1,
    )

    print(f"[INFO] Encoded Shape : {X_encoded.shape}")

    return X_encoded

def build_feature_names(
    encoders: dict,
) -> list[str]:
    """
    Build encoded feature names.
    """

    print("\n" + "=" * 60)
    print("BUILD FEATURE NAMES")
    print("=" * 60)

    feature_names = []

    # -------------------------
    # Team
    # -------------------------

    feature_names.extend(

        [
            f"team_{team}"

            for team in encoders["team"].categories_[0]

        ]

    )

    # -------------------------
    # Map
    # -------------------------

    feature_names.extend(

        [
            f"map_{map_name}"

            for map_name in encoders["map"].categories_[0]

        ]

    )

    # -------------------------
    # Meta
    # -------------------------

    feature_names.extend(

        [
            f"meta_{year}"

            for year in encoders["year"].categories_[0]

        ]

    )

    # -------------------------
    # Agent
    # -------------------------

    feature_names.extend(

        [
            f"agent_{agent}"

            for agent in encoders["agent"].classes_

        ]

    )

    print(f"[INFO] Total Feature : {len(feature_names)}")

    return feature_names

def validate_feature_names(
    X_encoded,
    feature_names,
):
    """
    Validate encoded feature names.
    """

    print("\n" + "=" * 60)
    print("VALIDATE FEATURE NAMES")
    print("=" * 60)

    if X_encoded.shape[1] != len(feature_names):

        raise ValueError(

            "Feature count mismatch.\n"

            f"Encoded : {X_encoded.shape[1]}\n"

            f"Names   : {len(feature_names)}"

        )

    print("[INFO] Validation Passed")
    
def build_feature_pipeline(
    df: pd.DataFrame,
) -> dict:
    """
    Complete feature engineering pipeline.
    """

    validate_dataset(df)

    X, y = split_feature_target(df)

    encoders = build_encoders(X)

    X_encoded = encode_features(
        X,
        encoders,
    )

    feature_names = build_feature_names(
        encoders
    )

    validate_feature_names(
        X_encoded,
        feature_names,
    )

    return {

        "X": X_encoded,

        "y": y,

        "encoders": encoders,

        "feature_names": feature_names,

    }
    
def debugging():
    df = load_dataset()
    
    validate_dataset(df)
    
    print()

    print("=" * 60)
    print("AGENT TYPE")
    print("=" * 60)

    print(type(df.iloc[0]["Agent"]))

    print(df.iloc[0]["Agent"])

    X, y = split_feature_target(df)
    
    encoders = build_encoders(X)
    
    X_encoded = encode_features(
        X,
        encoders,
    )
    
    print()

    print("=" * 60)
    print("ENCODED SAMPLE")
    print("=" * 60)

    print(X_encoded[:5])

    feature_names = build_feature_names(
        encoders
    )

    validate_feature_names(
        X_encoded,
        feature_names,
    )
    
    print()

    print("=" * 60)
    print("FEATURE NAME SAMPLE")
    print("=" * 60)

    for feature in feature_names[:10] :
        print(feature)
    
    print()

    print("=" * 60)
    print("FEATURE SAMPLE")
    print("=" * 60)

    print(X.head())

    print()

    print("=" * 60)
    print("TARGET SAMPLE")
    print("=" * 60)

    print(y.head())


def main():

    df = load_dataset()

    pipeline = build_feature_pipeline(df)

    print()

    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(f"X Shape : {pipeline['X'].shape}")

    print(f"y Shape : {pipeline['y'].shape}")

    print(f"Feature : {len(pipeline['feature_names'])}")

if __name__ == "__main__":
    main()