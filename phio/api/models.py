"""Pydantic models for API requests/responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    message: str = Field(..., description="Status message")


class PredictionRequest(BaseModel):
    """Prediction request model."""

    x: List[float] = Field(..., description="X coordinates")
    y: Optional[List[float]] = Field(None, description="Y coordinates")
    t: Optional[List[float]] = Field(None, description="Time coordinates")
    model_name: str = Field("default", description="Model name to use for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "x": [0.1, 0.2, 0.3, 0.4, 0.5],
                "y": [0.5, 0.5, 0.5, 0.5, 0.5],
                "t": [0.0, 0.0, 0.0, 0.0, 0.0],
                "model_name": "navier-stokes-re100",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""

    predictions: List[float] = Field(..., description="Model predictions")
    n_points: int = Field(..., description="Number of prediction points")
    model_version: str = Field(..., description="Model version used")


class TrainingRequest(BaseModel):
    """Training request model."""

    problem_type: str = Field(..., description="Problem type (heat, wave, navier-stokes)")
    domain: dict = Field(..., description="Spatial-temporal domain")
    n_epochs: int = Field(5000, description="Number of training epochs")
    learning_rate: float = Field(1e-3, description="Learning rate")
    architecture: dict = Field(
        {"hidden_dim": 128, "num_layers": 4},
        description="Network architecture",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "problem_type": "heat",
                "domain": {"x": [0, 1], "t": [0, 1]},
                "n_epochs": 5000,
                "learning_rate": 0.001,
                "architecture": {"hidden_dim": 128, "num_layers": 4},
            }
        }


class TrainingResponse(BaseModel):
    """Training response model."""

    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
