# api/main.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("my_lib.api")
logging.basicConfig(level=logging.INFO)


MODELS_DIR = Path("models")
LINEAR_COEF_PATH = MODELS_DIR / "linear_coef.npy"


class LinearPredictRequest(BaseModel):
    """Request body for linear regression predictions.

    Attributes
    ----------
    X : list[list[float]]
        2D array-like of shape (n_samples, n_features) containing the features.
    """

    X: list[list[float]]


class LinearPredictResponse(BaseModel):
    """Response body for linear regression predictions."""

    y_pred: list[float]


app = FastAPI(title="my_lib API", version="0.1.0")


def _load_linear_coef() -> np.ndarray:
    """Load the trained linear coefficients from disk.

    Returns
    -------
    coef : ndarray of shape (n_features + 1,)
        Coefficients with intercept as the first element.
    """
    if not LINEAR_COEF_PATH.exists():
        raise FileNotFoundError(
            f"Trained coefficients not found at {LINEAR_COEF_PATH}. "
            "Run scripts/train_linear.py (or `dvc repro`) first."
        )
    return np.load(LINEAR_COEF_PATH)


@app.post("/predict/linear", response_model=LinearPredictResponse)
def predict_linear(payload: LinearPredictRequest) -> LinearPredictResponse:
    """Predict using the trained linear regression model.

    Notes
    -----
    This expects that `scripts/train_linear.py` has been run at least once,
    so that `models/linear_coef.npy` exists on disk.
    """
    logger.info("Received /predict/linear request with %d samples", len(payload.X))
    # Convert input to ndarray
    X = np.asarray(payload.X, dtype=float)

    if X.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="X must be a 2D array-like of shape (n_samples, n_features).",
        )

    try:
        coef = _load_linear_coef()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # coef = [w0, w1, ..., w_p]
    # Model: y = w0 + X @ [w1, ..., w_p]
    intercept = coef[0]
    weights = coef[1:]

    if X.shape[1] != weights.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature dimension mismatch: got X.shape[1] = {X.shape[1]}, "
                f"but model expects {weights.shape[0]} features."
            ),
        )

    y_pred = intercept + X @ weights
    logger.info("Received /predict/linear request with %d samples", len(payload.X))
    return LinearPredictResponse(y_pred=y_pred.tolist())
