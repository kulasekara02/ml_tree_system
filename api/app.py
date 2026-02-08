"""
FastAPI REST API for ML Tree System.
Provides prediction endpoints with input validation.

Run with: uvicorn api.app:app --reload
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_artifact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_tree_api")

# Initialize FastAPI app
app = FastAPI(
    title="ML Tree System API",
    description="REST API for breast cancer classification using Decision Tree and Random Forest",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model storage
models: Dict[str, Any] = {}
preprocessor = None
metadata: Dict[str, Any] = {}


class FeatureInput(BaseModel):
    """
    Input schema for a single prediction.
    All 30 features from breast cancer dataset.
    """
    mean_radius: float = Field(..., ge=0, description="Mean of distances from center to points on the perimeter")
    mean_texture: float = Field(..., ge=0, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., ge=0, description="Mean perimeter")
    mean_area: float = Field(..., ge=0, description="Mean area")
    mean_smoothness: float = Field(..., ge=0, description="Local variation in radius lengths")
    mean_compactness: float = Field(..., ge=0, description="Perimeter^2 / area - 1.0")
    mean_concavity: float = Field(..., ge=0, description="Severity of concave portions")
    mean_concave_points: float = Field(..., ge=0, description="Number of concave portions")
    mean_symmetry: float = Field(..., ge=0, description="Symmetry")
    mean_fractal_dimension: float = Field(..., ge=0, description="Coastline approximation - 1")
    radius_error: float = Field(..., ge=0, alias="radius error")
    texture_error: float = Field(..., ge=0, alias="texture error")
    perimeter_error: float = Field(..., ge=0, alias="perimeter error")
    area_error: float = Field(..., ge=0, alias="area error")
    smoothness_error: float = Field(..., ge=0, alias="smoothness error")
    compactness_error: float = Field(..., ge=0, alias="compactness error")
    concavity_error: float = Field(..., ge=0, alias="concavity error")
    concave_points_error: float = Field(..., ge=0, alias="concave points error")
    symmetry_error: float = Field(..., ge=0, alias="symmetry error")
    fractal_dimension_error: float = Field(..., ge=0, alias="fractal dimension error")
    worst_radius: float = Field(..., ge=0, alias="worst radius")
    worst_texture: float = Field(..., ge=0, alias="worst texture")
    worst_perimeter: float = Field(..., ge=0, alias="worst perimeter")
    worst_area: float = Field(..., ge=0, alias="worst area")
    worst_smoothness: float = Field(..., ge=0, alias="worst smoothness")
    worst_compactness: float = Field(..., ge=0, alias="worst compactness")
    worst_concavity: float = Field(..., ge=0, alias="worst concavity")
    worst_concave_points: float = Field(..., ge=0, alias="worst concave points")
    worst_symmetry: float = Field(..., ge=0, alias="worst symmetry")
    worst_fractal_dimension: float = Field(..., ge=0, alias="worst fractal dimension")

    model_config = {"populate_by_name": True}


class SimplifiedFeatureInput(BaseModel):
    """
    Simplified input - accepts a list of 30 feature values.
    Easier for programmatic access.
    """
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="List of 30 feature values in order"
    )
    model_name: str = Field(
        default="random_forest",
        description="Model to use: 'decision_tree' or 'random_forest'"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v):
        if v not in ["decision_tree", "random_forest"]:
            raise ValueError("model_name must be 'decision_tree' or 'random_forest'")
        return v


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    samples: List[List[float]] = Field(
        ...,
        min_length=1,
        description="List of feature vectors (each with 30 values)"
    )
    model_name: str = Field(
        default="random_forest",
        description="Model to use"
    )


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    class_name: str = Field(..., description="Predicted class name")
    probability: float = Field(..., description="Probability of predicted class")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    model_used: str = Field(..., description="Model used for prediction")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[int]
    class_names: List[str]
    probabilities: List[Dict[str, float]]
    model_used: str
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    feature_names: List[str]
    target_names: List[str]
    best_params: Dict[str, Any]


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global models, preprocessor, metadata

    models_dir = Path("models")

    try:
        # Load Random Forest
        rf_path = models_dir / "random_forest.joblib"
        if rf_path.exists():
            models["random_forest"] = load_artifact(str(rf_path))
            logger.info("Loaded Random Forest model")

        # Load Decision Tree
        dt_path = models_dir / "decision_tree.joblib"
        if dt_path.exists():
            models["decision_tree"] = load_artifact(str(dt_path))
            logger.info("Loaded Decision Tree model")

        # Load preprocessor
        prep_path = models_dir / "preprocessor.joblib"
        if prep_path.exists():
            preprocessor = load_artifact(str(prep_path))
            logger.info("Loaded preprocessor")

        # Load metadata
        meta_path = models_dir / "metadata.joblib"
        if meta_path.exists():
            metadata = load_artifact(str(meta_path))
            logger.info("Loaded metadata")

        if not models:
            logger.warning("No models found. Run training first: python main.py train")

    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint - welcome message."""
    return {
        "message": "ML Tree System API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if models else "no models loaded",
        models_loaded=list(models.keys()),
        version="1.0.0"
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models."""
    return {
        "available_models": list(models.keys()),
        "default_model": "random_forest"
    }


@app.get("/models/{model_name}", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )

    best_params_key = f"best_params_{'dt' if model_name == 'decision_tree' else 'rf'}"

    return ModelInfoResponse(
        model_name=model_name,
        model_type=type(models[model_name]).__name__,
        feature_names=metadata.get("feature_names", []),
        target_names=metadata.get("target_names", []),
        best_params=metadata.get(best_params_key, {})
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: SimplifiedFeatureInput):
    """
    Make a single prediction.

    Input: 30 feature values as a list.
    Output: Predicted class and probabilities.
    """
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded. Run training first."
        )

    model_name = input_data.model_name
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' not found"
        )

    try:
        # Prepare input
        X = np.array(input_data.features).reshape(1, -1)

        # Preprocess
        if preprocessor:
            X = preprocessor.transform(X)

        # Predict
        model = models[model_name]
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        target_names = metadata.get("target_names", ["class_0", "class_1"])
        class_name = target_names[prediction]

        prob_dict = {name: float(prob) for name, prob in zip(target_names, probabilities)}

        return PredictionResponse(
            prediction=prediction,
            class_name=class_name,
            probability=float(probabilities[prediction]),
            probabilities=prob_dict,
            model_used=model_name
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(input_data: BatchInput):
    """
    Make batch predictions.

    Input: List of feature vectors.
    Output: List of predictions and probabilities.
    """
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded"
        )

    model_name = input_data.model_name
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' not found"
        )

    try:
        # Validate input dimensions
        for i, sample in enumerate(input_data.samples):
            if len(sample) != 30:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sample {i} has {len(sample)} features, expected 30"
                )

        # Prepare input
        X = np.array(input_data.samples)

        # Preprocess
        if preprocessor:
            X = preprocessor.transform(X)

        # Predict
        model = models[model_name]
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X)

        target_names = metadata.get("target_names", ["class_0", "class_1"])
        class_names = [target_names[p] for p in predictions]

        prob_dicts = [
            {name: float(prob) for name, prob in zip(target_names, probs)}
            for probs in probabilities
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            class_names=class_names,
            probabilities=prob_dicts,
            model_used=model_name,
            count=len(predictions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features", tags=["General"])
async def get_features():
    """Get expected feature names and order."""
    return {
        "feature_names": metadata.get("feature_names", []),
        "feature_count": len(metadata.get("feature_names", [])),
        "target_names": metadata.get("target_names", [])
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
