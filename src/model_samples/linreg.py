import numpy as np


class LinearRegression:
    def __init__(self, n_order: int = 1, n_max_iterations: int = 1000):
        """Linear regression using random optimization."""
        self.loss: list = []  # A history over loss values, so that you can plot your progress.
        self.n_order = n_order
        self._theta = np.zeros(shape=(n_order + 1, 1), dtype=np.float16)  # Size n_order + 1
        self.n_max_iterations = n_max_iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Does the training. X is a matrix with one data point per row, while y is flat."""
        for _ in range(self.n_max_iterations):
            self._fit(X, y)
            new_loss = self.score(X, y)
            self.loss.append(new_loss)
        
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Internal method that performs one iteration of the training. Should store a loss value in self.loss .""" 
        # Update
        old_theta = np.copy(self._theta)
        old_loss = self.score(X, y)
        self._theta = self._theta - np.random.normal(size=self._theta.shape) # Update theta randomly
        new_loss = self.score(X, y)

        if new_loss > old_loss:
            self._theta = old_theta

    def predict(self, X: np.ndarray): 
        """Predicts outputs y from some inputs X""" 
        data = X

        for i in range(self.n_order - 1):
            data = np.hstack((X, X ** (i + 2)))
        
        data = np.hstack((np.ones((X.shape[0], 1)), data))
        pred = np.dot(data, self._theta)

        return pred.flatten()

    def score(self, X, y): 
        """Returns the sum of squares residual, given inputs and outputs. """ 
        pred = self.predict(X)
        loss = sum([(y[i] - pred[i])**2 for i in range(len(y))]).item()
        return loss