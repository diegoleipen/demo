import numpy as np
from sklearn.linear_model import LogisticRegression
from my_lib.logistic_model import MyLogisticRegression


def make_binary_classification_data(
    n_samples: int = 200, n_features: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable-ish binary classification dataset."""
    rng = np.random.default_rng(seed)

    n_half = n_samples // 2
    X_pos = rng.normal(loc=1.0, scale=1.0, size=(n_half, n_features))
    X_neg = rng.normal(loc=-1.0, scale=1.0, size=(n_half, n_features))

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_half, dtype=int), np.zeros(n_half, dtype=int)])
    return X, y


def test_my_logistic_regression_performance_vs_sklearn() -> None:
    X_train, y_train = make_binary_classification_data(seed=0)

    sk = LogisticRegression(fit_intercept=True, solver="lbfgs")
    sk.fit(X_train, y_train)

    # Use more iterations so gradient descent converges nicely
    my = MyLogisticRegression(lr=0.1, n_iter=5000)
    my.fit(X_train, y_train)

    X_test, y_test = make_binary_classification_data(seed=1)

    sk_pred = sk.predict(X_test)
    my_pred = my.predict(X_test)

    sk_acc = np.mean(sk_pred == y_test)
    my_acc = np.mean(my_pred == y_test)

    # Your model should be pretty good on this simple dataset
    assert my_acc > 0.95
    # And not too far from sklearn's accuracy
    assert abs(my_acc - sk_acc) < 0.05


def test_logistic_regression_predict_before_fit_raises() -> None:
    import pytest

    model = MyLogisticRegression()
    X = np.zeros((5, 2))
    with pytest.raises(RuntimeError):
        model.predict(X)
