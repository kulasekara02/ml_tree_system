"""
Unit tests for data loading module.
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DataLoader(random_seed=42)

    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader(random_seed=123)
        assert loader.random_seed == 123
        assert loader.data is None
        assert loader.target is None

    def test_load_breast_cancer(self):
        """Test loading breast cancer dataset."""
        X, y = self.loader.load_sklearn_dataset("breast_cancer")

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 569  # Known dataset size
        assert len(X.columns) == 30  # Known feature count
        assert len(y) == 569
        assert set(y.unique()) == {0, 1}  # Binary classification

    def test_load_iris(self):
        """Test loading iris dataset."""
        X, y = self.loader.load_sklearn_dataset("iris")

        assert isinstance(X, pd.DataFrame)
        assert len(X) == 150
        assert len(X.columns) == 4
        assert set(y.unique()) == {0, 1, 2}  # 3 classes

    def test_load_invalid_dataset(self):
        """Test loading unknown dataset raises error."""
        with pytest.raises(ValueError) as excinfo:
            self.loader.load_sklearn_dataset("unknown_dataset")

        assert "Unknown dataset" in str(excinfo.value)

    def test_split_data(self):
        """Test data splitting."""
        self.loader.load_sklearn_dataset("breast_cancer")
        X_train, X_test, y_train, y_test = self.loader.split_data(test_size=0.2)

        # Check sizes
        assert len(X_train) == pytest.approx(569 * 0.8, abs=2)
        assert len(X_test) == pytest.approx(569 * 0.2, abs=2)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check no overlap
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx.intersection(test_idx)) == 0

    def test_split_without_loading(self):
        """Test splitting without loading data raises error."""
        with pytest.raises(ValueError) as excinfo:
            self.loader.split_data()

        assert "No data loaded" in str(excinfo.value)

    def test_data_summary(self):
        """Test data summary generation."""
        self.loader.load_sklearn_dataset("breast_cancer")
        summary = self.loader.get_data_summary()

        assert summary["n_samples"] == 569
        assert summary["n_features"] == 30
        assert len(summary["feature_names"]) == 30
        assert len(summary["target_names"]) == 2

    def test_data_quality_check(self):
        """Test data quality check."""
        self.loader.load_sklearn_dataset("breast_cancer")
        quality = self.loader.check_data_quality()

        assert "missing_by_column" in quality
        assert "duplicates" in quality
        assert "constant_columns" in quality

    def test_reproducibility(self):
        """Test that same seed gives same split."""
        loader1 = DataLoader(random_seed=42)
        loader1.load_sklearn_dataset("breast_cancer")
        X_train1, X_test1, y_train1, y_test1 = loader1.split_data(test_size=0.2)

        loader2 = DataLoader(random_seed=42)
        loader2.load_sklearn_dataset("breast_cancer")
        X_train2, X_test2, y_train2, y_test2 = loader2.split_data(test_size=0.2)

        # Should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
