import numpy as np


class LogisticRegression:
    def __init__(self, alpha: float = 0.1, n_max_iterations: int = 100):
        """Linear regression using gradient descent optimization for training."""
        self.loss: list = []    # A history over loss values, so that you can plot your progress.
        self.alpha = alpha      # learning rate
        self._theta = np.zeros(shape=(3,))  # 2 for variables, 1 for dummy. Hard code after the requirement of the assignment
        self.n_max_iterations = n_max_iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Does the training. X is a matrix with one data point per row, while y is flat."""

        # Run gradient descent n times
        for _ in range(self.n_max_iterations):
            self._fit(X, y)
            new_loss = self._calc_loss(X, y)
            self.loss.append(new_loss)
        
        return self
        
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Internal method that performs one iteration of the training. 
        At each iteration, calculate the gradient, and update theta
        .""" 
        old_theta = np.copy(self._theta)
        old_loss = self._calc_loss(X, y)

        # Update with gradient
        gradient = self._calc_gradient(X, y)
        self._theta = self._theta - self.alpha * gradient
        new_loss = self._calc_loss(X, y)

        if new_loss > old_loss:
            self._theta = old_theta
        
    def predict(self, X: np.ndarray): 
        """Predicts outputs y from some inputs X
        Convert the soft prediction to crisp label prediction by comparing soft_pred with a threshold,
        typically 0.5
        """ 
        soft_pred = self._soft_predict(X)
        pred = (soft_pred > 0.5) * 1

        return pred.ravel()

    def score(self, X: np.ndarray, y):
        """Return accuracy score: n correct labels / n total labels"""
        pred = self.predict(X)
        accuracy = sum(pred == y) / len(y)
        return accuracy

    def _calc_loss(self, X, y): 
        """Returns the sum of squares residual, given inputs and outputs. """ 
        pred = self.predict(X)
        loss = sum([(y[i] - pred[i])**2 for i in range(len(y))]).item()
        return loss
    
    def _sigmoid(self, input: np.ndarray):
        return 1 / (1 + np.e ** (-input))
    
    def _soft_predict(self, X: np.ndarray):
        """Output the sigmoid of dot product (data.T . theta)"""
        # Add dummy var to x
        data = np.hstack((np.ones((len(X), 1)), X))
        soft_pred = self._sigmoid(np.dot(data, self._theta)).ravel()
        return soft_pred
    
    def _calc_gradient(self, X: np.ndarray, y: np.ndarray):
        gradient_list = []
        data = np.hstack((np.ones((len(X), 1)), X))

        for i in range(self._theta.shape[0]):
            gradient = 1 / len(X) * sum(((self._soft_predict(X) - y) * data[:, i]).ravel())
            gradient_list.append(gradient)
            
        return np.asarray(gradient_list)