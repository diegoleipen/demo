# scripts/train_linear.py
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np

from my_lib.linear_model import MyLinearRegression


def make_regression_data(
    n_samples: int = 200,
    n_features: int = 3,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_coef = rng.normal(size=n_features + 1)
    y = true_coef[0] + X @ true_coef[1:] + noise * rng.normal(size=n_samples)
    return X, y


def main() -> None:
    # Hyperparameters
    lr = 0.1
    n_iter = 1000

    X_train, y_train = make_regression_data(seed=0)
    X_test, y_test = make_regression_data(seed=1)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("lr", lr)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("n_features", X_train.shape[1])

        model = MyLinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = float(np.mean((y_pred - y_test) ** 2))

        # Log metric
        mlflow.log_metric("mse", mse)

        print(f"MSE on test set: {mse:.4f}")

        # Save coefficients as a versioned model artifact
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        coef_path = models_dir / "linear_coef.npy"
        np.save(coef_path, model.coef_)

        # Log the same file to MLflow
        mlflow.log_artifact(str(coef_path))


if __name__ == "__main__":
    main()
