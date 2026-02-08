"""
Model training module with hyperparameter tuning.
Supports Decision Tree and Random Forest classifiers.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.base import BaseEstimator

logger = logging.getLogger("ml_tree_system")


class ModelTrainer:
    """
    Handles model training with hyperparameter optimization.
    Supports both grid search and random search.
    """

    MODEL_CLASSES = {
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
    }

    def __init__(
        self,
        model_type: str,
        random_seed: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the trainer.

        Args:
            model_type: Type of model (decision_tree, random_forest).
            random_seed: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all cores).
        """
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.MODEL_CLASSES.keys())}"
            )

        self.model_type = model_type
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self.model: Optional[BaseEstimator] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.search_object_: Optional[Union[GridSearchCV, RandomizedSearchCV]] = None

    def _create_base_model(self) -> BaseEstimator:
        """Create a base model instance with random seed."""
        model_class = self.MODEL_CLASSES[self.model_type]
        return model_class(random_state=self.random_seed)

    def train_with_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        search_type: str = "random",
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1"
    ) -> BaseEstimator:
        """
        Train model with hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training target.
            param_grid: Dictionary of parameters to search.
            search_type: 'grid' or 'random' search.
            n_iter: Number of iterations for random search.
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric for optimization.

        Returns:
            Best trained model.
        """
        base_model = self._create_base_model()

        # Configure cross-validation
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        logger.info(f"Starting {search_type} search for {self.model_type}")
        logger.info(f"Parameter grid: {param_grid}")

        if search_type == "grid":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                return_train_score=True
            )
        else:
            # Random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_seed,
                return_train_score=True
            )

        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.cv_results_ = search.cv_results_
        self.search_object_ = search

        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best CV score ({scoring}): {search.best_score_:.4f}")

        return self.model

    def train_simple(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **model_params
    ) -> BaseEstimator:
        """
        Train model with specified parameters (no tuning).

        Args:
            X_train: Training features.
            y_train: Training target.
            **model_params: Model-specific parameters.

        Returns:
            Trained model.
        """
        model_class = self.MODEL_CLASSES[self.model_type]
        self.model = model_class(
            random_state=self.random_seed,
            **model_params
        )

        logger.info(f"Training {self.model_type} with params: {model_params}")
        self.model.fit(X_train, y_train)

        return self.model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: Union[str, List[str]] = "accuracy"
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation on the trained model.

        Args:
            X: Features.
            y: Target.
            cv_folds: Number of folds.
            scoring: Scoring metric(s).

        Returns:
            Dictionary of CV scores per metric.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_* method first.")

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        if isinstance(scoring, str):
            scoring = [scoring]

        results = {}
        for metric in scoring:
            scores = cross_val_score(
                self.model,
                X, y,
                cv=cv,
                scoring=metric,
                n_jobs=self.n_jobs
            )
            results[metric] = scores
            logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        return results

    def get_feature_importance(
        self,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            feature_names: List of feature names.

        Returns:
            DataFrame with features and their importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        importance_df["cumulative"] = importance_df["importance"].cumsum()
        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)


def train_both_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dt_param_grid: Dict[str, List[Any]],
    rf_param_grid: Dict[str, List[Any]],
    random_seed: int = 42,
    search_type: str = "random",
    n_iter: int = 50,
    cv_folds: int = 5,
    scoring: str = "f1"
) -> Tuple[ModelTrainer, ModelTrainer]:
    """
    Train both Decision Tree and Random Forest models.

    Args:
        X_train: Training features.
        y_train: Training target.
        dt_param_grid: Decision Tree parameter grid.
        rf_param_grid: Random Forest parameter grid.
        random_seed: Random seed.
        search_type: Search type for tuning.
        n_iter: Iterations for random search.
        cv_folds: CV folds.
        scoring: Optimization metric.

    Returns:
        Tuple of (DecisionTree trainer, RandomForest trainer).
    """
    logger.info("=" * 50)
    logger.info("Training Decision Tree")
    logger.info("=" * 50)

    dt_trainer = ModelTrainer("decision_tree", random_seed=random_seed)
    dt_trainer.train_with_tuning(
        X_train, y_train,
        param_grid=dt_param_grid,
        search_type=search_type,
        n_iter=n_iter,
        cv_folds=cv_folds,
        scoring=scoring
    )

    logger.info("=" * 50)
    logger.info("Training Random Forest")
    logger.info("=" * 50)

    rf_trainer = ModelTrainer("random_forest", random_seed=random_seed)
    rf_trainer.train_with_tuning(
        X_train, y_train,
        param_grid=rf_param_grid,
        search_type=search_type,
        n_iter=n_iter,
        cv_folds=cv_folds,
        scoring=scoring
    )

    return dt_trainer, rf_trainer
