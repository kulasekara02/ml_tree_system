"""
Unit tests for model training and evaluation.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create synthetic classification data
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        self.feature_names = [f"feature_{i}" for i in range(10)]

    def test_trainer_init(self):
        """Test trainer initialization."""
        trainer = ModelTrainer("decision_tree", random_seed=42)

        assert trainer.model_type == "decision_tree"
        assert trainer.random_seed == 42
        assert trainer.model is None

    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError) as excinfo:
            ModelTrainer("invalid_model")

        assert "Unknown model type" in str(excinfo.value)

    def test_train_simple_dt(self):
        """Test simple training for Decision Tree."""
        trainer = ModelTrainer("decision_tree", random_seed=42)
        model = trainer.train_simple(self.X, self.y, max_depth=5)

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_simple_rf(self):
        """Test simple training for Random Forest."""
        trainer = ModelTrainer("random_forest", random_seed=42)
        model = trainer.train_simple(self.X, self.y, n_estimators=10, max_depth=5)

        assert model is not None
        assert hasattr(model, "predict")

    def test_predict(self):
        """Test prediction after training."""
        trainer = ModelTrainer("decision_tree", random_seed=42)
        trainer.train_simple(self.X, self.y)

        predictions = trainer.predict(self.X[:10])

        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self):
        """Test probability prediction."""
        trainer = ModelTrainer("random_forest", random_seed=42)
        trainer.train_simple(self.X, self.y, n_estimators=10)

        probabilities = trainer.predict_proba(self.X[:10])

        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_without_training(self):
        """Test prediction without training raises error."""
        trainer = ModelTrainer("decision_tree")

        with pytest.raises(ValueError) as excinfo:
            trainer.predict(self.X)

        assert "Model not trained" in str(excinfo.value)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        trainer = ModelTrainer("random_forest", random_seed=42)
        trainer.train_simple(self.X, self.y, n_estimators=10)

        importance_df = trainer.get_feature_importance(self.feature_names)

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 10
        assert importance_df["importance"].sum() == pytest.approx(1.0, abs=0.01)

    def test_cross_validation(self):
        """Test cross-validation."""
        trainer = ModelTrainer("decision_tree", random_seed=42)
        trainer.train_simple(self.X, self.y, max_depth=5)

        cv_results = trainer.cross_validate(
            self.X, self.y,
            cv_folds=3,
            scoring=["accuracy", "f1"]
        )

        assert "accuracy" in cv_results
        assert "f1" in cv_results
        assert len(cv_results["accuracy"]) == 3

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning with small grid."""
        trainer = ModelTrainer("decision_tree", random_seed=42)

        param_grid = {
            "max_depth": [3, 5],
            "min_samples_split": [2, 5]
        }

        model = trainer.train_with_tuning(
            self.X, self.y,
            param_grid=param_grid,
            search_type="grid",
            cv_folds=2
        )

        assert model is not None
        assert trainer.best_params_ is not None
        assert "max_depth" in trainer.best_params_

    def test_reproducibility(self):
        """Test training reproducibility with same seed."""
        trainer1 = ModelTrainer("random_forest", random_seed=42)
        trainer1.train_simple(self.X, self.y, n_estimators=10)
        pred1 = trainer1.predict(self.X[:10])

        trainer2 = ModelTrainer("random_forest", random_seed=42)
        trainer2.train_simple(self.X, self.y, n_estimators=10)
        pred2 = trainer2.predict(self.X[:10])

        np.testing.assert_array_equal(pred1, pred2)


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_prob = np.column_stack([
            1 - np.random.rand(10) * 0.3,
            np.random.rand(10) * 0.3
        ])
        # Adjust probabilities to match predictions better
        for i in range(len(self.y_pred)):
            if self.y_pred[i] == 1:
                self.y_prob[i] = [0.3, 0.7]
            else:
                self.y_prob[i] = [0.7, 0.3]

    def test_evaluate_classification(self):
        """Test classification metrics."""
        evaluator = ModelEvaluator(task_type="classification")
        metrics = evaluator.evaluate_classification(
            self.y_true,
            self.y_pred,
            self.y_prob,
            model_name="test_model"
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

        # Check metrics are in valid range
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classification(
            self.y_true,
            self.y_pred,
            model_name="test"
        )

        cm = metrics["confusion_matrix"]
        assert cm.shape == (2, 2)
        assert cm.sum() == len(self.y_true)

    def test_error_analysis(self):
        """Test error analysis."""
        evaluator = ModelEvaluator()
        X_test = np.random.randn(10, 5)
        feature_names = [f"f{i}" for i in range(5)]

        analysis = evaluator.error_analysis(
            X_test,
            self.y_true,
            self.y_pred,
            feature_names,
            n_samples=5
        )

        assert "total_errors" in analysis
        assert "error_rate" in analysis
        assert analysis["total_errors"] >= 0
        assert 0 <= analysis["error_rate"] <= 1

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = evaluator.evaluate_classification(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_classification_report(self):
        """Test classification report generation."""
        evaluator = ModelEvaluator()
        report = evaluator.get_classification_report(
            self.y_true,
            self.y_pred,
            target_names=["class_0", "class_1"]
        )

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
