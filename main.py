"""
ML Tree System - Main Entry Point
Command-line interface for training, evaluation, and prediction.

Usage:
    python main.py train --config config/config.yaml
    python main.py predict --model models/random_forest.joblib --input data/sample.csv
    python main.py evaluate --model models/random_forest.joblib
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers

from src.utils.helpers import (
    load_config,
    setup_logging,
    set_seeds,
    save_artifact,
    load_artifact,
    ensure_dir
)
from src.data.data_loader import DataLoader
from src.features.preprocessing import PreprocessingPipeline
from src.models.trainer import ModelTrainer, train_both_models
from src.models.evaluator import ModelEvaluator, create_comparison_table
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_roc_comparison,
    plot_model_comparison
)


def train_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the full training pipeline.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with trained models and results.
    """
    logger = logging.getLogger("ml_tree_system")
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Set random seed
    seed = config.get("random_seed", 42)
    set_seeds(seed)
    logger.info(f"Random seed set to: {seed}")

    # Create directories
    models_dir = ensure_dir(config["paths"]["models_dir"])
    figures_dir = ensure_dir(config["paths"]["figures_dir"])

    # Step 1: Load data
    logger.info("\n--- Step 1: Loading Data ---")
    data_loader = DataLoader(random_seed=seed)
    dataset_name = config["data"]["dataset_name"]
    X, y = data_loader.load_sklearn_dataset(dataset_name)

    # Data summary
    summary = data_loader.get_data_summary()
    logger.info(f"Dataset summary: {summary['n_samples']} samples, {summary['n_features']} features")

    # Quality check
    quality = data_loader.check_data_quality()

    # Step 2: Split data
    logger.info("\n--- Step 2: Splitting Data ---")
    X_train, X_test, y_train, y_test = data_loader.split_data(
        test_size=config["data"]["test_size"]
    )

    # Step 3: Preprocessing
    logger.info("\n--- Step 3: Preprocessing ---")
    preprocessor = PreprocessingPipeline(
        scaler_type=config["preprocessing"]["scaler"],
        imputer_strategy=config["preprocessing"]["handle_missing"]
    )
    preprocessor.build_pipeline()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Step 4: Train models
    logger.info("\n--- Step 4: Training Models ---")
    dt_trainer, rf_trainer = train_both_models(
        X_train_processed,
        y_train.values,
        dt_param_grid=config["decision_tree"]["param_grid"],
        rf_param_grid=config["random_forest"]["param_grid"],
        random_seed=seed,
        search_type=config["tuning"]["search_type"],
        n_iter=config["tuning"]["n_iter"],
        cv_folds=config["cross_validation"]["n_folds"],
        scoring=config["tuning"]["scoring"]
    )

    # Step 5: Evaluate models
    logger.info("\n--- Step 5: Evaluating Models ---")
    comparison_df = create_comparison_table(
        dt_trainer, rf_trainer,
        X_test_processed, y_test.values
    )

    # Detailed evaluation
    evaluator = ModelEvaluator(task_type="classification")

    dt_pred = dt_trainer.predict(X_test_processed)
    dt_prob = dt_trainer.predict_proba(X_test_processed)
    rf_pred = rf_trainer.predict(X_test_processed)
    rf_prob = rf_trainer.predict_proba(X_test_processed)

    # Error analysis
    logger.info("\n--- Error Analysis ---")
    dt_errors = evaluator.error_analysis(
        X_test_processed, y_test.values, dt_pred,
        data_loader.feature_names
    )
    rf_errors = evaluator.error_analysis(
        X_test_processed, y_test.values, rf_pred,
        data_loader.feature_names
    )

    # Step 6: Visualizations
    logger.info("\n--- Step 6: Generating Visualizations ---")

    # Confusion matrices
    plot_confusion_matrix(
        y_test.values, dt_pred,
        class_names=data_loader.target_names,
        title="Decision Tree - Confusion Matrix",
        save_path=str(figures_dir / "confusion_matrix_dt.png")
    )
    plot_confusion_matrix(
        y_test.values, rf_pred,
        class_names=data_loader.target_names,
        title="Random Forest - Confusion Matrix",
        save_path=str(figures_dir / "confusion_matrix_rf.png")
    )

    # Feature importance
    rf_importance = rf_trainer.get_feature_importance(data_loader.feature_names)
    plot_feature_importance(
        rf_importance,
        top_n=15,
        title="Random Forest - Feature Importance",
        save_path=str(figures_dir / "feature_importance_rf.png")
    )

    dt_importance = dt_trainer.get_feature_importance(data_loader.feature_names)
    plot_feature_importance(
        dt_importance,
        top_n=15,
        title="Decision Tree - Feature Importance",
        save_path=str(figures_dir / "feature_importance_dt.png")
    )

    # Decision tree visualization
    plot_decision_tree(
        dt_trainer.model,
        feature_names=data_loader.feature_names,
        class_names=data_loader.target_names,
        max_depth=3,
        title="Decision Tree Structure (depth=3)",
        save_path=str(figures_dir / "decision_tree.png")
    )

    # ROC comparison
    plot_roc_comparison(
        {
            "Decision Tree": dt_prob[:, 1],
            "Random Forest": rf_prob[:, 1]
        },
        y_test.values,
        save_path=str(figures_dir / "roc_comparison.png")
    )

    # Model comparison bar chart
    plot_model_comparison(
        comparison_df,
        metrics=["Accuracy", "Precision", "Recall", "F1-Score"],
        title="Model Performance Comparison",
        save_path=str(figures_dir / "model_comparison.png")
    )

    # Step 7: Save artifacts
    logger.info("\n--- Step 7: Saving Artifacts ---")

    # Save models
    save_artifact(dt_trainer.model, str(models_dir / "decision_tree.joblib"))
    save_artifact(rf_trainer.model, str(models_dir / "random_forest.joblib"))
    logger.info("Saved trained models")

    # Save preprocessing pipeline
    save_artifact(preprocessor.pipeline, str(models_dir / "preprocessor.joblib"))
    logger.info("Saved preprocessing pipeline")

    # Save feature names and target names for inference
    metadata = {
        "feature_names": data_loader.feature_names,
        "target_names": data_loader.target_names,
        "best_params_dt": dt_trainer.best_params_,
        "best_params_rf": rf_trainer.best_params_,
    }
    save_artifact(metadata, str(models_dir / "metadata.joblib"))
    logger.info("Saved metadata")

    # Save comparison table
    comparison_df.to_csv(str(models_dir / "model_comparison.csv"), index=False)

    # Save feature importance
    rf_importance.to_csv(str(models_dir / "feature_importance.csv"), index=False)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest Decision Tree Params: {dt_trainer.best_params_}")
    print(f"Best Random Forest Params: {rf_trainer.best_params_}")
    print(f"\nArtifacts saved to: {models_dir}")
    print(f"Figures saved to: {figures_dir}")

    return {
        "dt_trainer": dt_trainer,
        "rf_trainer": rf_trainer,
        "preprocessor": preprocessor,
        "comparison": comparison_df,
        "metadata": metadata
    }


def predict_pipeline(
    model_path: str,
    input_data: pd.DataFrame,
    preprocessor_path: str,
    metadata_path: str
) -> np.ndarray:
    """
    Make predictions using a trained model.

    Args:
        model_path: Path to trained model.
        input_data: Input features as DataFrame.
        preprocessor_path: Path to preprocessing pipeline.
        metadata_path: Path to metadata file.

    Returns:
        Array of predictions.
    """
    logger = logging.getLogger("ml_tree_system")

    # Load artifacts
    model = load_artifact(model_path)
    preprocessor = load_artifact(preprocessor_path)
    metadata = load_artifact(metadata_path)

    logger.info(f"Loaded model from: {model_path}")

    # Ensure correct feature order
    feature_names = metadata["feature_names"]

    # Handle missing features
    missing = set(feature_names) - set(input_data.columns)
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    # Reorder columns
    input_data = input_data[feature_names]

    # Preprocess
    X_processed = preprocessor.transform(input_data)

    # Predict
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)

    # Map to class names
    target_names = metadata["target_names"]
    predicted_classes = [target_names[p] for p in predictions]

    logger.info(f"Made {len(predictions)} predictions")

    return predictions, probabilities, predicted_classes


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="ML Tree System - Decision Tree and Random Forest Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --config config/config.yaml
  python main.py predict --model models/random_forest.joblib --input data/sample.csv
  python main.py api --port 8000
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model"
    )
    predict_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    predict_parser.add_argument(
        "--preprocessor", "-p",
        type=str,
        default="models/preprocessor.joblib",
        help="Path to preprocessor"
    )
    predict_parser.add_argument(
        "--metadata",
        type=str,
        default="models/metadata.joblib",
        help="Path to metadata"
    )
    predict_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path"
    )

    # API command
    api_parser = subparsers.add_parser("api", help="Start REST API server")
    api_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logger = setup_logging(level="INFO")

    if args.command == "train":
        # Load config and run training
        config = load_config(args.config)
        train_pipeline(config)

    elif args.command == "predict":
        # Load input data
        input_data = pd.read_csv(args.input)
        logger.info(f"Loaded {len(input_data)} samples from {args.input}")

        # Make predictions
        predictions, probabilities, classes = predict_pipeline(
            args.model,
            input_data,
            args.preprocessor,
            args.metadata
        )

        # Output results
        results_df = input_data.copy()
        results_df["prediction"] = predictions
        results_df["predicted_class"] = classes
        results_df["probability_0"] = probabilities[:, 0]
        results_df["probability_1"] = probabilities[:, 1]

        if args.output:
            results_df.to_csv(args.output, index=False)
            logger.info(f"Results saved to: {args.output}")
        else:
            print("\nPrediction Results:")
            print(results_df[["prediction", "predicted_class", "probability_1"]].to_string())

    elif args.command == "api":
        # Import and run API
        import uvicorn
        from api.app import app

        logger.info(f"Starting API server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
