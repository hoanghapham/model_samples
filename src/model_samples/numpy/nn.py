import numpy as np


class FeedForwardNeuralNetwork:
    def __init__(
            self, 
            n_input_units: int = 3, 
            n_hidden: int = 10,
            alpha: float = 0.001,
            eta0: int = 3,
            n_max_iterations: int = 10000
        ) -> None:
        self.n_input_units = n_input_units
        self.n_hidden = n_hidden
        self._theta1: np.ndarray = np.random.normal(size=(n_input_units, n_hidden))
        self._theta2: np.ndarray = np.random.normal(size=(n_hidden + 1, 1))   # binary classification only need 1 output unit
        self.n_max_iterations = n_max_iterations
        self.loss: list = []
        self.eta0 = eta0        # learning rate
        self.alpha = alpha      # Regularization
        self.best_loss = float('inf')

        
    @property
    def theta_(self):
        return np.concatenate((self._theta1.ravel(), self._theta2.ravel()))

    @theta_.setter
    def theta_(self, value: np.ndarray):
        self._theta1 = value[:self._theta1.size].reshape(self._theta1.shape)
        self._theta2 = value[self._theta1.size:].reshape(self._theta2.shape)
    
    def _fit(self, X, y):
        """just for compliance"""
        pass

    def fit(self, X, y):
        best_loss = float('inf')
        
        for i in range(self.n_max_iterations):
            current_theta = self.theta_.copy()
            self.theta_ = self.theta_ - np.random.normal(size=self.theta_.shape)
            current_loss = self._loss(X, y)
            self.loss.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                print(f"Iter {i + 1}, training loss: {current_loss:.4f}", end="\r")
            else:
                self.theta_ = current_theta

            # Gradient descent - update theta
            # Add regularizatioin here
            # l2_reg = self._calc_l2_reg()
            # update_rule = np.asarray([[0,1,1] * self.n_input_units + [0,1,1] * 1]).ravel()  # number of output units
            
        return self

    def _soft_predict(self, X: np.ndarray):
        """Output the sigmoid of dot product (data.T . theta)"""
        # Add dummy var to x
        data = np.hstack((np.ones((len(X), 1)), X))
        activation1 = self._sigmoid(np.dot(data, self._theta1))
        # print(activation1.shape)
        activation1 = np.hstack((np.ones((activation1.shape[0], 1)), activation1))
        soft_pred = self._sigmoid(np.dot(activation1, self._theta2))
        return soft_pred
    
    def _calc_l2_reg(self):
        regu = self.alpha * np.sum(self.theta_ ** 2)
        return regu

    def _loss(self, X: np.ndarray, y: np.ndarray):
        """Calculate logistic loss (cross entropy loss)"""
        soft_pred = self._soft_predict(X)
        soft_pred[y == 0] = 1 - soft_pred[y == 0]
        l2_reg = self._calc_l2_reg()
        objective = np.sum(-np.log(soft_pred)) + l2_reg
        return objective

    def _calc_gradient(self, X, y):
        soft_pred = self._soft_predict(X)
        gradient = 1 / len(X) * np.sum(np.dot((soft_pred - y), X))
        return gradient

    def _sigmoid(self, value: np.ndarray):
        return 1 / (1 + np.e ** (-value))

    def predict(self, X):
        """Predicts outputs y from some inputs X
        Convert the soft prediction to crisp label prediction by comparing soft_pred with a threshold,
        typically 0.5
        """ 
        soft_pred = self._soft_predict(X)
        pred = (soft_pred > 0.5) * 1

        return pred.ravel()

    def score(self, X, y):
        """Return accuracy score: n correct labels / n total labels"""
        pred = self.predict(X)
        accuracy = sum(pred == y) / len(y)
        return accuracy

#TODO
# Do not regularize bias
# calculate correct gradient