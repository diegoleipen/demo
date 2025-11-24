import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from my_lib.linear_model import MyLinearRegression


def make_regression_data(
    n_samples: int = 100, n_features: int = 3, noise: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple linear regression dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_coef = rng.normal(size=n_features + 1)  # bias + weights
    y = true_coef[0] + X @ true_coef[1:] + noise * rng.normal(size=n_samples)
    return X, y, true_coef


def test_my_linear_regression_matches_sklearn() -> None:
    X_train, y_train, _ = make_regression_data(noise=0.1, seed=0)

    sk = LinearRegression(fit_intercept=True)
    sk.fit(X_train, y_train)

    my = MyLinearRegression()
    my.fit(X_train, y_train)

    # compare predictions on a separate test set
    X_test, y_test, _ = make_regression_data(n_samples=50, noise=0.1, seed=1)
    sk_pred = sk.predict(X_test)
    my_pred = my.predict(X_test)

    # They should be numerically very close
    assert np.allclose(my_pred, sk_pred, rtol=1e-5, atol=1e-5)


def test_linear_regression_predict_before_fit_raises() -> None:
    model = MyLinearRegression()
    X = np.zeros((5, 2))
    with pytest.raises(RuntimeError):
        model.predict(X)
