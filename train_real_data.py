"""
Training script for real public datasets with GPU support.

Usage:
    python train_real_data.py --dataset heart_disease --gpu
    python train_real_data.py --dataset adult_income --no-gpu
    python train_real_data.py --dataset diabetes --gpu
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.utils.helpers import (
    load_config,
    setup_logging,
    set_seeds,
    save_artifact,
    ensure_dir
)
from src.data.dataset_downloader import DatasetDownloader
from src.features.preprocessing import PreprocessingPipeline
from src.models.gpu_trainer import GPUModelTrainer, train_models_gpu, check_gpu_availability
from src.models.evaluator import ModelEvaluator
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_roc_comparison,
    plot_model_comparison
)
from sklearn.model_selection import train_test_split


def train_real_dataset(
    dataset_name: str,
    use_gpu: bool = True,
    config_path: str = "config/config.yaml"
) -> Dict[str, Any]:
    """
    Train models on a real public dataset.

    Args:
        dataset_name: Name of dataset to use
        use_gpu: Whether to use GPU acceleration
        config_path: Path to config file

    Returns:
        Dictionary with results
    """
    # Setup
    logger = setup_logging(level="INFO")
    config = load_config(config_path)
    seed = config.get("random_seed", 42)
    set_seeds(seed)

    # Directories
    models_dir = ensure_dir("models")
    figures_dir = ensure_dir("reports/figures")

    logger.info("=" * 60)
    logger.info(f"TRAINING ON REAL DATASET: {dataset_name.upper()}")
    logger.info("=" * 60)

    # Check GPU
    gpu_info = check_gpu_availability()
    logger.info(f"GPU Available: {gpu_info['gpu_available']}")
    if gpu_info['gpu_available']:
        logger.info(f"GPU: {gpu_info.get('gpu_name', 'Unknown')}")

    # Step 1: Download and load data
    logger.info("\n--- Step 1: Loading Real Dataset ---")
    downloader = DatasetDownloader(data_dir="data/raw")
    X, y, metadata = downloader.get_dataset(dataset_name)

    logger.info(f"Dataset: {metadata['description']}")
    logger.info(f"Samples: {metadata['n_samples']}")
    logger.info(f"Features: {metadata['n_features']}")
    logger.info(f"Feature names: {metadata['feature_names'][:5]}...")

    # Step 2: Split data
    logger.info("\n--- Step 2: Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=seed,
        stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

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

    dt_trainer, rf_trainer = train_models_gpu(
        X_train_processed,
        y_train.values,
        dt_param_grid=config["decision_tree"]["param_grid"],
        rf_param_grid=config["random_forest"]["param_grid"],
        use_gpu=use_gpu,
        random_seed=seed,
        search_type=config["tuning"]["search_type"],
        n_iter=config["tuning"]["n_iter"],
        cv_folds=config["cross_validation"]["n_folds"],
        scoring=config["tuning"]["scoring"]
    )

    # Step 5: Evaluate
    logger.info("\n--- Step 5: Evaluating Models ---")
    evaluator = ModelEvaluator(task_type="classification")

    dt_pred = dt_trainer.predict(X_test_processed)
    dt_prob = dt_trainer.predict_proba(X_test_processed)
    rf_pred = rf_trainer.predict(X_test_processed)
    rf_prob = rf_trainer.predict_proba(X_test_processed)

    # Metrics
    dt_metrics = evaluator.evaluate_classification(
        y_test.values, dt_pred, dt_prob, model_name="Decision Tree"
    )
    rf_metrics = evaluator.evaluate_classification(
        y_test.values, rf_pred, rf_prob, model_name="Random Forest"
    )

    # Comparison table
    comparison_df = pd.DataFrame([
        {
            "Model": "Random Forest",
            "Accuracy": rf_metrics["accuracy"],
            "Precision": rf_metrics["precision"],
            "Recall": rf_metrics["recall"],
            "F1-Score": rf_metrics["f1"],
            "ROC-AUC": rf_metrics.get("roc_auc", np.nan)
        },
        {
            "Model": "Decision Tree",
            "Accuracy": dt_metrics["accuracy"],
            "Precision": dt_metrics["precision"],
            "Recall": dt_metrics["recall"],
            "F1-Score": dt_metrics["f1"],
            "ROC-AUC": dt_metrics.get("roc_auc", np.nan)
        }
    ]).sort_values("F1-Score", ascending=False)

    # Error analysis
    logger.info("\n--- Error Analysis ---")
    dt_errors = evaluator.error_analysis(
        X_test_processed, y_test.values, dt_pred,
        metadata["feature_names"]
    )
    rf_errors = evaluator.error_analysis(
        X_test_processed, y_test.values, rf_pred,
        metadata["feature_names"]
    )

    # Step 6: Visualizations
    logger.info("\n--- Step 6: Generating Visualizations ---")

    target_names = metadata["target_names"]

    # Confusion matrices
    plot_confusion_matrix(
        y_test.values, dt_pred,
        class_names=target_names,
        title=f"Decision Tree - {dataset_name}",
        save_path=str(figures_dir / f"cm_dt_{dataset_name}.png")
    )
    plot_confusion_matrix(
        y_test.values, rf_pred,
        class_names=target_names,
        title=f"Random Forest - {dataset_name}",
        save_path=str(figures_dir / f"cm_rf_{dataset_name}.png")
    )

    # Feature importance
    rf_importance = rf_trainer.get_feature_importance(metadata["feature_names"])
    plot_feature_importance(
        rf_importance,
        top_n=min(15, len(metadata["feature_names"])),
        title=f"Feature Importance - {dataset_name}",
        save_path=str(figures_dir / f"importance_{dataset_name}.png")
    )

    # ROC comparison
    plot_roc_comparison(
        {
            "Decision Tree": dt_prob[:, 1],
            "Random Forest": rf_prob[:, 1]
        },
        y_test.values,
        save_path=str(figures_dir / f"roc_{dataset_name}.png")
    )

    # Model comparison
    plot_model_comparison(
        comparison_df,
        metrics=["Accuracy", "Precision", "Recall", "F1-Score"],
        title=f"Model Comparison - {dataset_name}",
        save_path=str(figures_dir / f"comparison_{dataset_name}.png")
    )

    # Decision tree visualization (if not too large)
    if hasattr(dt_trainer.model, "tree_") and dt_trainer.model.tree_.max_depth <= 10:
        try:
            plot_decision_tree(
                dt_trainer.model,
                feature_names=metadata["feature_names"],
                class_names=target_names,
                max_depth=3,
                title=f"Decision Tree - {dataset_name}",
                save_path=str(figures_dir / f"tree_{dataset_name}.png")
            )
        except Exception as e:
            logger.warning(f"Could not plot tree: {e}")

    # Step 7: Save artifacts
    logger.info("\n--- Step 7: Saving Artifacts ---")

    save_artifact(dt_trainer.model, str(models_dir / f"dt_{dataset_name}.joblib"))
    save_artifact(rf_trainer.model, str(models_dir / f"rf_{dataset_name}.joblib"))
    save_artifact(preprocessor.pipeline, str(models_dir / f"preprocessor_{dataset_name}.joblib"))
    save_artifact(metadata, str(models_dir / f"metadata_{dataset_name}.joblib"))
    comparison_df.to_csv(str(models_dir / f"comparison_{dataset_name}.csv"), index=False)
    rf_importance.to_csv(str(models_dir / f"importance_{dataset_name}.csv"), index=False)

    logger.info("Artifacts saved")

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {dataset_name.upper()}")
    print("=" * 60)
    print(f"\nDataset: {metadata['description']}")
    print(f"Samples: {metadata['n_samples']} | Features: {metadata['n_features']}")
    print(f"GPU Used: {dt_trainer.is_gpu_model or rf_trainer.is_gpu_model}")
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest DT Params: {dt_trainer.best_params_}")
    print(f"Best RF Params: {rf_trainer.best_params_}")
    print(f"\nArtifacts: {models_dir}")
    print(f"Figures: {figures_dir}")

    return {
        "dt_trainer": dt_trainer,
        "rf_trainer": rf_trainer,
        "comparison": comparison_df,
        "metadata": metadata
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train on real public datasets with GPU support"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="heart_disease",
        choices=["heart_disease", "adult_income", "wine_quality", "diabetes"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU only"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all available datasets"
    )

    args = parser.parse_args()

    use_gpu = args.gpu and not args.no_gpu

    if args.all:
        datasets = ["heart_disease", "adult_income", "wine_quality", "diabetes"]
        for dataset in datasets:
            print(f"\n{'#' * 60}")
            print(f"# DATASET: {dataset}")
            print(f"{'#' * 60}")
            train_real_dataset(dataset, use_gpu=use_gpu, config_path=args.config)
    else:
        train_real_dataset(args.dataset, use_gpu=use_gpu, config_path=args.config)


if __name__ == "__main__":
    main()
