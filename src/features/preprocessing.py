"""
Preprocessing and feature engineering module.
Builds sklearn pipelines for data transformation.
"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger("ml_tree_system")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific features.
    Useful for dropping low-importance or constant features.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Args:
            columns: List of column names to keep. None keeps all.
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """Fit - just validates columns exist."""
        if self.columns is not None:
            missing = set(self.columns) - set(X.columns)
            if missing:
                logger.warning(f"Columns not found in data: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select specified columns."""
        if self.columns is None:
            return X
        available = [c for c in self.columns if c in X.columns]
        return X[available]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers using IQR method.
    Values beyond Q1 - factor*IQR or Q3 + factor*IQR are clipped.
    """

    def __init__(self, factor: float = 1.5):
        """
        Args:
            factor: IQR multiplier for outlier bounds.
        """
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Calculate outlier bounds from training data."""
        X_arr = np.array(X)
        Q1 = np.percentile(X_arr, 25, axis=0)
        Q3 = np.percentile(X_arr, 75, axis=0)
        IQR = Q3 - Q1

        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Clip values to calculated bounds."""
        X_arr = np.array(X)
        X_clipped = np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
        return X_clipped


class PreprocessingPipeline:
    """
    Factory class to build preprocessing pipelines.
    Handles missing values, scaling, and feature engineering.
    """

    SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    IMPUTERS = {
        "mean": "mean",
        "median": "median",
        "most_frequent": "most_frequent",
    }

    def __init__(
        self,
        scaler_type: str = "standard",
        imputer_strategy: str = "median",
        clip_outliers: bool = False,
        outlier_factor: float = 1.5
    ):
        """
        Initialize preprocessing configuration.

        Args:
            scaler_type: Type of scaler (standard, minmax, robust).
            imputer_strategy: Strategy for missing values (mean, median, most_frequent).
            clip_outliers: Whether to clip outliers.
            outlier_factor: IQR factor for outlier clipping.
        """
        self.scaler_type = scaler_type
        self.imputer_strategy = imputer_strategy
        self.clip_outliers = clip_outliers
        self.outlier_factor = outlier_factor
        self.pipeline: Optional[Pipeline] = None

    def build_pipeline(
        self,
        numeric_features: Optional[List[str]] = None
    ) -> Pipeline:
        """
        Build the preprocessing pipeline.

        Args:
            numeric_features: List of numeric column names.

        Returns:
            Configured sklearn Pipeline.
        """
        steps = []

        # Step 1: Handle missing values
        imputer = SimpleImputer(strategy=self.IMPUTERS.get(
            self.imputer_strategy, "median"
        ))
        steps.append(("imputer", imputer))

        # Step 2: Outlier clipping (optional)
        if self.clip_outliers:
            clipper = OutlierClipper(factor=self.outlier_factor)
            steps.append(("outlier_clipper", clipper))

        # Step 3: Scaling
        scaler_class = self.SCALERS.get(self.scaler_type, StandardScaler)
        steps.append(("scaler", scaler_class()))

        self.pipeline = Pipeline(steps)

        logger.info(f"Built preprocessing pipeline: {[s[0] for s in steps]}")

        return self.pipeline

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit the pipeline on training data and transform.

        Args:
            X_train: Training features.
            y_train: Training target (unused, for API consistency).

        Returns:
            Transformed training data.
        """
        if self.pipeline is None:
            self.build_pipeline()

        logger.info("Fitting and transforming training data")
        X_transformed = self.pipeline.fit_transform(X_train)

        return X_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.

        Args:
            X: Data to transform.

        Returns:
            Transformed data.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call fit_transform first.")

        return self.pipeline.transform(X)

    def get_feature_names(self, input_features: List[str]) -> List[str]:
        """
        Get output feature names after transformation.

        Args:
            input_features: Original feature names.

        Returns:
            Transformed feature names (same for numeric pipeline).
        """
        # For our numeric-only pipeline, names stay the same
        return input_features


def create_full_preprocessing_pipeline(
    scaler: str = "standard",
    imputer: str = "median",
    clip_outliers: bool = False
) -> PreprocessingPipeline:
    """
    Factory function to create a preprocessing pipeline.

    Args:
        scaler: Scaler type.
        imputer: Imputer strategy.
        clip_outliers: Whether to clip outliers.

    Returns:
        Configured PreprocessingPipeline instance.
    """
    pipeline = PreprocessingPipeline(
        scaler_type=scaler,
        imputer_strategy=imputer,
        clip_outliers=clip_outliers
    )
    return pipeline
