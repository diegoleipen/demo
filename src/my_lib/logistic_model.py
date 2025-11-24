from __future__ import annotations

import logging
from typing import cast

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray


class MyLogisticRegression(BaseModel):
    """
    Binary logistic regression with gradient descent optimization.

    This model estimates the probability of the positive class using::

        p(y = 1 | x) = sigmoid(beta_0 + beta_1 x_1 + ... + beta_p x_p)

    where ``sigmoid(z) = 1 / (1 + exp(-z))``.

    Examples
    --------
    Learn a simple decision boundary and predict on new points::

        >>> import numpy as np
        >>> from my_lib.logistic_model import MyLogisticRegression
        >>> X = np.array([[-1.0], [1.0]])
        >>> y = np.array([0, 1])
        >>> model = MyLogisticRegression(lr=0.5, n_iter=1000)
        >>> _ = model.fit(X, y)
        >>> preds = model.predict(np.array([[-2.0], [2.0]]))
        >>> preds.tolist()
        [0, 1]
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000) -> None:
        """
        Parameters
        ----------
        lr : float, default=0.1
            Learning rate for gradient descent.
        n_iter : int, default=1000
            Number of gradient descent iterations.
        """
        self.lr: float = lr
        self.n_iter: int = n_iter
        self.coef_: ArrayLike | None = None

    @staticmethod
    def _sigmoid(z: ArrayLike) -> ArrayLike:
        """Numerically stable sigmoid function."""
        # Clip to avoid overflow in exp for large |z|
        z = np.clip(z, -500, 500)
        return cast(ArrayLike, 1.0 / (1.0 + np.exp(-z)))

    def fit(self, X: ArrayLike, y: ArrayLike) -> MyLogisticRegression:
        """
        Fit the logistic regression model using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples,)
            Binary target labels in {0, 1}.

        Returns
        -------
        MyLogisticRegression
            The fitted model instance.

        Examples
        --------
        >>> import numpy as np
        >>> from my_lib.logistic_model import MyLogisticRegression
        >>> X = np.array([[0.0], [1.0], [2.0]])
        >>> y = np.array([0, 0, 1])
        >>> model = MyLogisticRegression(lr=0.5, n_iter=2000)
        >>> _ = model.fit(X, y)
        >>> isinstance(model.coef_, np.ndarray)
        True
        """
        logger.info("Fitting MyLogisticRegression")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")

        n_samples, n_features = X.shape
        X_design = np.c_[np.ones((n_samples, 1)), X]

        self.coef_ = np.zeros(X_design.shape[1], dtype=float)

        for _ in range(self.n_iter):
            logits = X_design @ self.coef_
            probs = self._sigmoid(logits)
            grad = X_design.T @ (probs - y) / n_samples
            self.coef_ -= self.lr * grad

        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict binary class labels for the given input.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels in {0, 1}.

        Examples
        --------
        >>> import numpy as np
        >>> from my_lib.logistic_model import MyLogisticRegression
        >>> X = np.array([[-1.0], [1.0]])
        >>> y = np.array([0, 1])
        >>> model = MyLogisticRegression(lr=0.5, n_iter=1000)
        >>> _ = model.fit(X, y)
        >>> preds = model.predict(np.array([[-2.0], [2.0]]))
        >>> preds.tolist()
        [0, 1]
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        X_design = np.c_[np.ones((X.shape[0], 1)), X]
        probs = self._sigmoid(X_design @ self.coef_)
        return (probs >= 0.5).astype(int)
