"""FastAPI application for PhIO."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import jax.numpy as jnp
import numpy as np

from phio.api.models import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="PhIO API",
        description="Physics-Informed Neural Network Inference API",
        version="0.3.2",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_model=HealthResponse)
    async def root():
        """Root endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.3.2",
            message="PhIO API is running",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.3.2",
            message="All systems operational",
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """PINN prediction endpoint.

        Args:
            request: Prediction request with input coordinates

        Returns:
            Predictions at requested points
        """
        try:
            # Convert input to JAX arrays
            x = jnp.array(request.x)
            y = jnp.array(request.y) if request.y is not None else None
            t = jnp.array(request.t) if request.t is not None else None

            # TODO: Load model and make predictions
            # For now, return dummy prediction
            if y is not None and t is not None:
                # 2D+time (Navier-Stokes)
                u = np.sin(x * np.pi) * np.cos(y * np.pi) * np.exp(-t)
                predictions = u.tolist()
            elif t is not None:
                # 1D+time (Heat equation)
                u = np.sin(x * np.pi) * np.exp(-t)
                predictions = u.tolist()
            else:
                raise ValueError("Invalid input dimensions")

            return PredictionResponse(
                predictions=predictions,
                n_points=len(predictions),
                model_version="demo-v1",
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
