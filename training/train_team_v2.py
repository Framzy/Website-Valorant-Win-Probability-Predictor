"""
Valorant Predictor V2
Training Entry Point
"""

from training_utils import (
    aggregate_matches,
    calculate_target,
    cross_validate_model,
    encode_features,
    load_dataset,
    log_info,
    log_section,
    print_feature_importance,
    save_artifacts,
    save_model_metadata,
    save_training_summary,
    train_model,
    validate_dataset,
)


def main():

    log_section(
        "VALORANT PREDICTOR V2 TRAINING"
    )

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------
    # Aggregate Composition
    # --------------------------------------------------

    grouped = aggregate_matches(df)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    validate_dataset(grouped)

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    grouped = calculate_target(grouped)

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    X, y, artifacts = encode_features(
        grouped
    )

    log_info(
        "Feature engineering complete."
    )
    
    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    model, metrics = train_model(
        X,
        y,
    )

    # --------------------------------------------------
    # Cross Validation
    # --------------------------------------------------

    cv_metrics = cross_validate_model(
        X,
        y,
    )

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    print_feature_importance(

        model,

        artifacts["feature_columns"],

    )

    # --------------------------------------------------
    # Save Artifacts
    # --------------------------------------------------

    save_artifacts(

        model,

        artifacts,

    )

    # --------------------------------------------------
    # Save Metadata
    # --------------------------------------------------

    save_model_metadata(

        metrics,

        cv_metrics,

        X,

    )

    # --------------------------------------------------
    # Save Summary
    # --------------------------------------------------

    save_training_summary(

        grouped,

    )

    log_section(
        "TRAINING FINISHED"
    )

    log_info(
        "All artifacts saved successfully."
    )
    
if __name__ == "__main__":

    main()