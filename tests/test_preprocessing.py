"""
Unit tests for preprocessing module.
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.preprocessing import (
    PreprocessingPipeline,
    FeatureSelector,
    OutlierClipper,
    create_full_preprocessing_pipeline
)


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100) * 10,
            "feature_3": np.random.randn(100) + 5,
        })
        self.X_test = pd.DataFrame({
            "feature_1": np.random.randn(20),
            "feature_2": np.random.randn(20) * 10,
            "feature_3": np.random.randn(20) + 5,
        })

    def test_pipeline_init(self):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(
            scaler_type="standard",
            imputer_strategy="median"
        )

        assert pipeline.scaler_type == "standard"
        assert pipeline.imputer_strategy == "median"
        assert pipeline.pipeline is None

    def test_build_pipeline(self):
        """Test pipeline building."""
        preprocessor = PreprocessingPipeline()
        pipeline = preprocessor.build_pipeline()

        assert pipeline is not None
        assert len(pipeline.steps) >= 2  # At least imputer and scaler

    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = PreprocessingPipeline()
        preprocessor.build_pipeline()
        X_transformed = preprocessor.fit_transform(self.X_train)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == self.X_train.shape

        # Check standardization (mean ~0, std ~1)
        assert np.abs(X_transformed.mean()) < 0.1
        assert np.abs(X_transformed.std() - 1.0) < 0.1

    def test_transform(self):
        """Test transform on new data."""
        preprocessor = PreprocessingPipeline()
        preprocessor.build_pipeline()
        preprocessor.fit_transform(self.X_train)

        X_test_transformed = preprocessor.transform(self.X_test)

        assert isinstance(X_test_transformed, np.ndarray)
        assert X_test_transformed.shape == self.X_test.shape

    def test_transform_before_fit(self):
        """Test transform without fit raises error."""
        preprocessor = PreprocessingPipeline()

        with pytest.raises(ValueError) as excinfo:
            preprocessor.transform(self.X_test)

        assert "Pipeline not built" in str(excinfo.value)

    def test_different_scalers(self):
        """Test different scaler types."""
        for scaler_type in ["standard", "minmax", "robust"]:
            preprocessor = PreprocessingPipeline(scaler_type=scaler_type)
            preprocessor.build_pipeline()
            X_transformed = preprocessor.fit_transform(self.X_train)

            assert X_transformed.shape == self.X_train.shape

    def test_with_missing_values(self):
        """Test handling of missing values."""
        # Add missing values
        X_with_nan = self.X_train.copy()
        X_with_nan.iloc[0, 0] = np.nan
        X_with_nan.iloc[5, 1] = np.nan

        preprocessor = PreprocessingPipeline(imputer_strategy="median")
        preprocessor.build_pipeline()
        X_transformed = preprocessor.fit_transform(X_with_nan)

        # Should have no NaN after preprocessing
        assert not np.isnan(X_transformed).any()


class TestOutlierClipper:
    """Tests for OutlierClipper class."""

    def test_outlier_clipping(self):
        """Test that outliers are clipped."""
        # Create data with outliers
        X = np.array([[1, 2], [1.5, 2.5], [100, -100], [2, 3]])

        clipper = OutlierClipper(factor=1.5)
        X_clipped = clipper.fit_transform(X)

        # Extreme values should be clipped
        assert X_clipped.max() < 100
        assert X_clipped.min() > -100

    def test_normal_data_unchanged(self):
        """Test that normal data is mostly unchanged."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        clipper = OutlierClipper(factor=3.0)  # Wide bounds
        X_clipped = clipper.fit_transform(X)

        # Most values should be unchanged with wide factor
        np.testing.assert_array_almost_equal(X, X_clipped, decimal=5)


class TestFeatureSelector:
    """Tests for FeatureSelector class."""

    def test_select_columns(self):
        """Test column selection."""
        X = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9]
        })

        selector = FeatureSelector(columns=["a", "c"])
        X_selected = selector.fit_transform(X)

        assert list(X_selected.columns) == ["a", "c"]
        assert len(X_selected.columns) == 2

    def test_select_all(self):
        """Test selecting all columns (None)."""
        X = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })

        selector = FeatureSelector(columns=None)
        X_selected = selector.fit_transform(X)

        pd.testing.assert_frame_equal(X, X_selected)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_pipeline(self):
        """Test factory function creates valid pipeline."""
        pipeline = create_full_preprocessing_pipeline(
            scaler="minmax",
            imputer="mean",
            clip_outliers=True
        )

        assert isinstance(pipeline, PreprocessingPipeline)
        assert pipeline.scaler_type == "minmax"
        assert pipeline.imputer_strategy == "mean"
        assert pipeline.clip_outliers is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
