"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def random_seed():
    """Provide consistent random seed."""
    return 42


@pytest.fixture(scope="session")
def sample_classification_data():
    """Generate sample classification data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y, feature_names


@pytest.fixture(scope="session")
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_1": np.random.randn(50),
        "feature_2": np.random.randn(50) * 10,
        "feature_3": np.random.randn(50) + 5,
    })


@pytest.fixture
def breast_cancer_sample():
    """Generate a sample similar to breast cancer dataset."""
    return [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
        184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
