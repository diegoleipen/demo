# src/my_lib/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsArray(Protocol):
    def __array__(self, dtype: Any | None = None) -> np.ndarray: ...


ArrayLike = np.ndarray  # you can refine this later


class BaseModel(ABC):
    """Abstract base class for simple ML models."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseModel:
        """Fit the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict using the model."""
        raise NotImplementedError
