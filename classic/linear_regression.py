import typing as tp
import numpy as np
from copy import deepcopy


class LinearRegression:
    def __init__(self,
                 lr: float,
                 iterations: tp.Optional[int] = None,
                 epsilon: tp.Optional[float] = None):
        self.w = None
        self.b = None
        self.lr = lr
        self.iterations = iterations
        self.epsilon = epsilon

        self.rows = None
        self.cols = None

    def _init_weights(self, cols: int) -> None:
        self.w = np.zeros(shape=(cols,))
        self.b = 0

    def fit(self, X: np.array, y: np.array) -> None:
        self.rows, self.cols = X.shape
        self._train(X, y)

    def _train(self, X: np.array, y: np.array) -> None:
        self._init_weights(self.cols)
        self._update_weights(X, y)

    def predict(self, X: np.array) -> np.array:
        return np.dot(X, self.w) + self.b

    def _update_in_iterations(self, X: np.array, y_true: np.array) -> None:
        for _ in range(self.iterations):
            self._update(X, y_true)

    def _update(self, X: np.array, y_true: np.array) -> None:
        y_pred = self.predict(X)
        new_w = - 2 * X.T.dot(y_true - y_pred) / self.rows
        self.w -= self.lr * new_w

        new_b = - 2 * np.sum(y_true - y_pred) / self.rows
        self.b -= self.lr * new_b

    def _get_epsilon(self) -> float:
        return np.linalg.norm(self.w)

    def _update_by_epsilon(self, X: np.array, y_true: np.array) -> None:
        prev_w = deepcopy(self.w)
        cur_w = deepcopy(self.w + 100)
        while np.linalg.norm(cur_w - prev_w) > self.epsilon:
            prev_w = deepcopy(cur_w)
            self._update(X, y_true)
            cur_w = deepcopy(self.w)

    def _update_weights(self, X: np.array, y_true: np.array) -> None:
        if self.iterations is not None:
            self._update_in_iterations(X, y_true)
        elif self.epsilon is not None:
            self._update_by_epsilon(X, y_true)
        else:
            raise Exception("You must choose one method")


class RidgeLR(LinearRegression):
    def __init__(self,
                 lr: float,
                 reg_lr: float,
                 iterations: tp.Optional[int] = None,
                 epsilon: tp.Optional[float] = None,
                 ):
        super().__init__(lr, iterations, epsilon)
        self.reg_lr = reg_lr

    def _update(self, X: np.array, y_true: np.array) -> None:
        y_pred = self.predict(X)
        new_w = - 2 * X.T.dot(y_true - y_pred) / self.rows
        self.w -= self.lr * new_w + 2 * self.reg_lr * self.w

        new_b = - 2 * np.sum(y_true - y_pred) / self.rows
        self.b -= self.lr * new_b


class LassoLR(LinearRegression):
    def __init__(self,
                 lr: float,
                 reg_lr: float,
                 iterations: tp.Optional[int] = None,
                 epsilon: tp.Optional[float] = None,
                 ):
        super().__init__(lr, iterations, epsilon)
        self.reg_lr = reg_lr

    def _update(self, X: np.array, y_true: np.array) -> None:
        y_pred = self.predict(X)
        new_w = - 2 * X.T.dot(y_true - y_pred) / self.rows
        self.w -= (
                      self.lr * new_w
                      + 2 * self.reg_lr * np.where(self.w > 0, 1, -1)
        )

        new_b = - 2 * np.sum(y_true - y_pred) / self.rows
        self.b -= self.lr * new_b


if __name__ == "__main__":
    X = np.random.normal(0, 1, size=(100, 4))
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.8 * X[:, 3]
    linregs = [
        LinearRegression(lr=0.1, epsilon=1e-8),
        RidgeLR(lr=0.1, epsilon=1e-8, reg_lr=0.0001),
        LassoLR(lr=0.1, epsilon=1e-8, reg_lr=0.0001),
        LinearRegression(lr=0.1, iterations=100),
        RidgeLR(lr=0.1, iterations=100, reg_lr=0.0001),
        LassoLR(lr=0.1, iterations=100, reg_lr=0.0001)
    ]
    for model in linregs:
        model.fit(X, y)
        print(model.w)
