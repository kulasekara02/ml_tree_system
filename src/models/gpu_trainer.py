"""
GPU-accelerated model training using RAPIDS cuML.
Falls back to scikit-learn if GPU not available.
"""
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger("ml_tree_system")

# Try to import GPU libraries
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    import cudf
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.tree import DecisionTreeClassifier as cuDT
    from cuml.model_selection import train_test_split as cu_train_test_split
    GPU_AVAILABLE = True
    CUML_AVAILABLE = True
    logger.info("RAPIDS cuML detected - GPU acceleration enabled")
except ImportError:
    logger.info("RAPIDS cuML not found - using CPU (scikit-learn)")

# Always import sklearn as fallback
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class GPUModelTrainer:
    """
    GPU-accelerated model trainer with automatic fallback to CPU.
    Uses RAPIDS cuML when available, scikit-learn otherwise.
    """

    def __init__(
        self,
        model_type: str,
        use_gpu: bool = True,
        random_seed: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize trainer.

        Args:
            model_type: 'decision_tree' or 'random_forest'
            use_gpu: Whether to attempt GPU acceleration
            random_seed: Random seed for reproducibility
            n_jobs: CPU parallel jobs (ignored for GPU)
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self.model = None
        self.best_params_ = None
        self.is_gpu_model = False

        if self.use_gpu:
            logger.info(f"GPU mode enabled for {model_type}")
        else:
            logger.info(f"CPU mode for {model_type}")

    def _get_gpu_model(self, **params):
        """Get GPU model instance."""
        if self.model_type == "decision_tree":
            return cuDT(random_state=self.random_seed, **params)
        elif self.model_type == "random_forest":
            return cuRF(random_state=self.random_seed, **params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_cpu_model(self, **params):
        """Get CPU model instance."""
        if self.model_type == "decision_tree":
            return DecisionTreeClassifier(random_state=self.random_seed, **params)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                **params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _to_gpu(self, X: np.ndarray, y: np.ndarray = None):
        """Convert numpy arrays to GPU arrays."""
        if not self.use_gpu:
            return X, y

        X_gpu = cp.asarray(X, dtype=cp.float32)
        if y is not None:
            y_gpu = cp.asarray(y, dtype=cp.int32)
            return X_gpu, y_gpu
        return X_gpu, None

    def train_gpu(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **model_params
    ):
        """
        Train model on GPU.

        Args:
            X_train: Training features
            y_train: Training labels
            **model_params: Model hyperparameters
        """
        if not self.use_gpu:
            return self.train_cpu(X_train, y_train, **model_params)

        logger.info(f"Training {self.model_type} on GPU...")

        # Convert to GPU arrays
        X_gpu, y_gpu = self._to_gpu(X_train, y_train)

        # Create and train model
        self.model = self._get_gpu_model(**model_params)
        self.model.fit(X_gpu, y_gpu)
        self.is_gpu_model = True

        logger.info("GPU training complete")
        return self.model

    def train_cpu(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **model_params
    ):
        """
        Train model on CPU.

        Args:
            X_train: Training features
            y_train: Training labels
            **model_params: Model hyperparameters
        """
        logger.info(f"Training {self.model_type} on CPU...")

        self.model = self._get_cpu_model(**model_params)
        self.model.fit(X_train, y_train)
        self.is_gpu_model = False

        logger.info("CPU training complete")
        return self.model

    def train_with_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        search_type: str = "random",
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1"
    ):
        """
        Train with hyperparameter tuning.
        Note: Tuning uses CPU (sklearn) for cross-validation compatibility.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter search space
            search_type: 'grid' or 'random'
            n_iter: Iterations for random search
            cv_folds: Cross-validation folds
            scoring: Optimization metric
        """
        logger.info(f"Hyperparameter tuning for {self.model_type}")
        logger.info(f"Search type: {search_type}, CV folds: {cv_folds}")

        # Use CPU model for tuning (sklearn CV compatibility)
        base_model = self._get_cpu_model()

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_seed
        )

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

        self.best_params_ = search.best_params_
        logger.info(f"Best params: {self.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        # Retrain final model (GPU if available)
        if self.use_gpu:
            self.train_gpu(X_train, y_train, **self.best_params_)
        else:
            self.model = search.best_estimator_
            self.is_gpu_model = False

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.is_gpu_model and self.use_gpu:
            X_gpu = cp.asarray(X, dtype=cp.float32)
            preds = self.model.predict(X_gpu)
            return cp.asnumpy(preds)
        else:
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.is_gpu_model and self.use_gpu:
            X_gpu = cp.asarray(X, dtype=cp.float32)
            proba = self.model.predict_proba(X_gpu)
            return cp.asnumpy(proba)
        else:
            return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")

        importances = self.model.feature_importances_

        if self.is_gpu_model:
            importances = cp.asnumpy(importances)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        importance_df["cumulative"] = importance_df["importance"].cumsum()
        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return system info.

    Returns:
        Dictionary with GPU status and info
    """
    info = {
        "gpu_available": GPU_AVAILABLE,
        "cuml_available": CUML_AVAILABLE,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory": None
    }

    if GPU_AVAILABLE:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            info["gpu_name"] = cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode()
            info["gpu_memory"] = f"{device.mem_info[1] / 1e9:.1f} GB"
            info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")

    return info


def train_models_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dt_param_grid: Dict[str, List[Any]],
    rf_param_grid: Dict[str, List[Any]],
    use_gpu: bool = True,
    random_seed: int = 42,
    search_type: str = "random",
    n_iter: int = 50,
    cv_folds: int = 5,
    scoring: str = "f1"
) -> Tuple[GPUModelTrainer, GPUModelTrainer]:
    """
    Train both models with optional GPU acceleration.

    Args:
        X_train: Training features
        y_train: Training labels
        dt_param_grid: Decision Tree params
        rf_param_grid: Random Forest params
        use_gpu: Enable GPU if available
        random_seed: Random seed
        search_type: Tuning search type
        n_iter: Random search iterations
        cv_folds: CV folds
        scoring: Optimization metric

    Returns:
        Tuple of (dt_trainer, rf_trainer)
    """
    # Check GPU
    gpu_info = check_gpu_availability()
    if use_gpu and gpu_info["gpu_available"]:
        logger.info(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']})")
    else:
        logger.info("Running on CPU")

    # Train Decision Tree
    logger.info("=" * 50)
    logger.info("Training Decision Tree")
    logger.info("=" * 50)

    dt_trainer = GPUModelTrainer(
        "decision_tree",
        use_gpu=use_gpu,
        random_seed=random_seed
    )
    dt_trainer.train_with_tuning(
        X_train, y_train,
        param_grid=dt_param_grid,
        search_type=search_type,
        n_iter=n_iter,
        cv_folds=cv_folds,
        scoring=scoring
    )

    # Train Random Forest
    logger.info("=" * 50)
    logger.info("Training Random Forest")
    logger.info("=" * 50)

    rf_trainer = GPUModelTrainer(
        "random_forest",
        use_gpu=use_gpu,
        random_seed=random_seed
    )
    rf_trainer.train_with_tuning(
        X_train, y_train,
        param_grid=rf_param_grid,
        search_type=search_type,
        n_iter=n_iter,
        cv_folds=cv_folds,
        scoring=scoring
    )

    return dt_trainer, rf_trainer
