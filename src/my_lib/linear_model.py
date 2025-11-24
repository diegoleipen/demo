# src/my_lib/linear_model.py
from __future__ import annotations

import logging
from typing import Final, cast

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray  # keep it simple for now


class MyLinearRegression(BaseModel):
    """
    Simple linear regression using the normal equations.

    This model fits a linear relationship of the form::

        y = beta_0 + beta_1 x_1 + ... + beta_p x_p

    using the closed-form solution:

    .. math::

        \\beta = (X^T X)^{-1} X^T y

    Examples
    --------
    Fit a simple linear function and compare predictions::
        >>> import numpy as np
        >>> from my_lib.linear_model import MyLinearRegression
        >>> X = np.array([[1.0], [2.0], [3.0]])
        >>> y = np.array([2.0, 4.0, 6.0])
        >>> model = MyLinearRegression().fit(X, y)
        >>> X_test = np.array([[4.0]])
        >>> import numpy as np
        >>> np.allclose(model.predict(X_test), np.array([8.0]))
        True
    """

    _BIAS_COLUMN: Final[float] = 1.0

    def __init__(self) -> None:
        self.coef_: ArrayLike | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> MyLinearRegression:
        """
        Fit the linear regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        MyLinearRegression
            The fitted model instance.

        Examples
        --------
        >>> import numpy as np
        >>> from my_lib.linear_model import MyLinearRegression
        >>> X = np.array([[0.0], [1.0]])
        >>> y = np.array([0.0, 1.0])
        >>> model = MyLinearRegression().fit(X, y)
        >>> isinstance(model.coef_, np.ndarray)
        True
        """
        logger.info("Fitting MyLinearRegression")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")

        # Design matrix with bias term
        X_design = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal equation using pseudo-inverse
        self.coef_ = np.linalg.pinv(X_design) @ y
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target values for the given input.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.

        Examples
        --------
        >>> import numpy as np
        >>> from my_lib.linear_model import MyLinearRegression
        >>> X = np.array([[1.0], [2.0], [3.0]])
        >>> y = np.array([2.0, 4.0, 6.0])
        >>> model = MyLinearRegression().fit(X, y)
        >>> preds = model.predict(np.array([[4.0], [5.0]]))
        >>> preds.shape
        (2,)
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        X_design = np.c_[np.ones((X.shape[0], 1)), X]
        return cast(ArrayLike, X_design @ self.coef_)
