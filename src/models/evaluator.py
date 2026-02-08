"""
Model evaluation module.
Computes metrics, generates comparison tables, and performs error analysis.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.base import BaseEstimator

logger = logging.getLogger("ml_tree_system")


class ModelEvaluator:
    """
    Comprehensive model evaluation for classification and regression.
    """

    def __init__(self, task_type: str = "classification"):
        """
        Initialize evaluator.

        Args:
            task_type: 'classification' or 'regression'.
        """
        self.task_type = task_type
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "model",
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Prediction probabilities (for ROC-AUC).
            model_name: Name for logging/storage.
            average: Averaging method for multi-class.

        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        # ROC-AUC (requires probabilities)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average=average
                    )
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = np.nan

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        self.results[model_name] = metrics

        logger.info(f"\n{model_name} Metrics:")
        for key, value in metrics.items():
            if key != "confusion_matrix":
                logger.info(f"  {key}: {value:.4f}")

        return metrics

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Compute regression metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            model_name: Name for logging/storage.

        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }

        self.results[model_name] = metrics

        logger.info(f"\n{model_name} Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return metrics

    def compare_models(
        self,
        models: Dict[str, Tuple[BaseEstimator, np.ndarray, np.ndarray, Optional[np.ndarray]]],
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple models and create a comparison table.

        Args:
            models: Dict of {name: (model, y_pred, y_prob)}.
            y_test: True test labels.

        Returns:
            DataFrame comparing all models.
        """
        comparison_data = []

        for name, (model, y_pred, y_prob) in models.items():
            if self.task_type == "classification":
                metrics = self.evaluate_classification(
                    y_test, y_pred, y_prob, model_name=name
                )
                row = {
                    "Model": name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1"],
                    "ROC-AUC": metrics.get("roc_auc", np.nan),
                }
            else:
                metrics = self.evaluate_regression(y_test, y_pred, model_name=name)
                row = {
                    "Model": name,
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "R2": metrics["r2"],
                }

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1 (classification) or R2 (regression)
        sort_col = "F1-Score" if self.task_type == "classification" else "R2"
        comparison_df = comparison_df.sort_values(sort_col, ascending=False)

        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        return comparison_df

    def error_analysis(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        n_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors to understand model failures.

        Args:
            X_test: Test features.
            y_true: True labels.
            y_pred: Predicted labels.
            feature_names: List of feature names.
            n_samples: Number of error samples to return.

        Returns:
            Dictionary with error analysis results.
        """
        # Find misclassified samples
        errors_mask = y_true != y_pred
        error_indices = np.where(errors_mask)[0]

        analysis = {
            "total_errors": int(errors_mask.sum()),
            "error_rate": float(errors_mask.mean()),
            "error_indices": error_indices.tolist()[:n_samples],
        }

        if len(error_indices) > 0:
            # Create DataFrame of errors
            error_df = pd.DataFrame(
                X_test[errors_mask][:n_samples],
                columns=feature_names
            )
            error_df["true_label"] = y_true[errors_mask][:n_samples]
            error_df["predicted"] = y_pred[errors_mask][:n_samples]
            analysis["error_samples"] = error_df

            # Error distribution by true class
            error_by_class = pd.Series(y_true[errors_mask]).value_counts()
            analysis["errors_by_true_class"] = error_by_class.to_dict()

            # Feature statistics for errors vs correct
            correct_mask = ~errors_mask
            if correct_mask.sum() > 0:
                error_means = X_test[errors_mask].mean(axis=0)
                correct_means = X_test[correct_mask].mean(axis=0)
                feature_diff = pd.DataFrame({
                    "feature": feature_names,
                    "error_mean": error_means,
                    "correct_mean": correct_means,
                    "difference": error_means - correct_means
                }).sort_values("difference", key=abs, ascending=False)
                analysis["feature_difference"] = feature_diff

        logger.info(f"\nError Analysis:")
        logger.info(f"  Total errors: {analysis['total_errors']}")
        logger.info(f"  Error rate: {analysis['error_rate']:.2%}")
        if "errors_by_true_class" in analysis:
            logger.info(f"  Errors by class: {analysis['errors_by_true_class']}")

        return analysis

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Get detailed classification report.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            target_names: Class names.

        Returns:
            Formatted classification report string.
        """
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )
        return report


def create_comparison_table(
    dt_trainer,
    rf_trainer,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Create a comparison table for Decision Tree and Random Forest.

    Args:
        dt_trainer: Trained Decision Tree trainer.
        rf_trainer: Trained Random Forest trainer.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Comparison DataFrame.
    """
    evaluator = ModelEvaluator(task_type="classification")

    dt_pred = dt_trainer.predict(X_test)
    dt_prob = dt_trainer.predict_proba(X_test)

    rf_pred = rf_trainer.predict(X_test)
    rf_prob = rf_trainer.predict_proba(X_test)

    models = {
        "Decision Tree": (dt_trainer.model, dt_pred, dt_prob),
        "Random Forest": (rf_trainer.model, rf_pred, rf_prob),
    }

    return evaluator.compare_models(models, y_test)
