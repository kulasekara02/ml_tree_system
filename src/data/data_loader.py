"""
Data loading and initial exploration module.
Handles loading from sklearn datasets or CSV files.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ml_tree_system")


class DataLoader:
    """
    Handles data loading from various sources.
    Supports sklearn built-in datasets and CSV files.
    """

    SKLEARN_DATASETS = {
        "breast_cancer": load_breast_cancer,
        "iris": load_iris,
        "wine": load_wine,
    }

    def __init__(self, random_seed: int = 42):
        """
        Initialize DataLoader.

        Args:
            random_seed: Seed for reproducible train/test splits.
        """
        self.random_seed = random_seed
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.feature_names: Optional[list] = None
        self.target_names: Optional[list] = None

    def load_sklearn_dataset(self, name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a built-in sklearn dataset.

        Args:
            name: Dataset name (breast_cancer, iris, wine).

        Returns:
            Tuple of (features DataFrame, target Series).

        Raises:
            ValueError: If dataset name is not recognized.
        """
        if name not in self.SKLEARN_DATASETS:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available: {list(self.SKLEARN_DATASETS.keys())}"
            )

        logger.info(f"Loading sklearn dataset: {name}")
        dataset = self.SKLEARN_DATASETS[name]()

        self.feature_names = list(dataset.feature_names)
        self.target_names = list(dataset.target_names)

        self.data = pd.DataFrame(
            dataset.data,
            columns=self.feature_names
        )
        self.target = pd.Series(dataset.target, name="target")

        logger.info(f"Loaded {len(self.data)} samples with {len(self.feature_names)} features")
        logger.info(f"Target classes: {self.target_names}")

        return self.data, self.target

    def load_csv(
        self,
        filepath: str,
        target_column: str,
        drop_columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to CSV file.
            target_column: Name of the target column.
            drop_columns: Columns to exclude from features.

        Returns:
            Tuple of (features DataFrame, target Series).
        """
        logger.info(f"Loading CSV from: {filepath}")

        df = pd.read_csv(filepath)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        self.target = df[target_column]
        columns_to_drop = [target_column]

        if drop_columns:
            columns_to_drop.extend(drop_columns)

        self.data = df.drop(columns=columns_to_drop, errors="ignore")
        self.feature_names = list(self.data.columns)
        self.target_names = list(self.target.unique())

        logger.info(f"Loaded {len(self.data)} samples with {len(self.feature_names)} features")

        return self.data, self.target

    def split_data(
        self,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            test_size: Fraction of data for testing.
            stratify: Whether to stratify split by target.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if self.data is None or self.target is None:
            raise ValueError("No data loaded. Call load_* method first.")

        stratify_target = self.target if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.target,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_target
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the loaded data.

        Returns:
            Dictionary with data statistics.
        """
        if self.data is None:
            return {}

        summary = {
            "n_samples": len(self.data),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "missing_values": self.data.isnull().sum().sum(),
            "target_distribution": self.target.value_counts().to_dict(),
            "dtypes": self.data.dtypes.value_counts().to_dict(),
        }

        return summary

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Check for common data quality issues.

        Returns:
            Dictionary with quality check results.
        """
        if self.data is None:
            return {}

        quality_report = {
            "missing_by_column": self.data.isnull().sum().to_dict(),
            "duplicates": self.data.duplicated().sum(),
            "constant_columns": [
                col for col in self.data.columns
                if self.data[col].nunique() == 1
            ],
            "high_cardinality": [
                col for col in self.data.columns
                if self.data[col].nunique() > 50 and self.data[col].dtype == "object"
            ],
        }

        # Log quality issues
        total_missing = sum(quality_report["missing_by_column"].values())
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values")
        if quality_report["duplicates"] > 0:
            logger.warning(f"Found {quality_report['duplicates']} duplicate rows")
        if quality_report["constant_columns"]:
            logger.warning(f"Found constant columns: {quality_report['constant_columns']}")

        return quality_report
