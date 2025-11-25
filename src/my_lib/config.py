from __future__ import annotations

from pydantic import BaseModel, Field


class LinearConfig(BaseModel):
    """Hyperparameters for MyLinearRegression."""

    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Learning rate for gradient-based methods (not used yet).",
    )
    n_iter: int = Field(
        default=1000,
        gt=0,
        description="Number of iterations for gradient-based methods (not used yet).",
    )
    fit_intercept: bool = True
