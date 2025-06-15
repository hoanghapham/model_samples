import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable
from sklearn.metrics import precision_recall_fscore_support


def get_sign(array: np.ndarray) -> np.ndarray:
    sign = array.copy()
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign[sign == 0] = 0
    return sign


def create_batches(iterable: Iterable, size=1):
    """Create batches of equal size from an iterable"""
    l = len(iterable)
    for idx in range(0, l, size):
        yield iterable[idx:min(idx + size, l)]


class SimpleSGDClassifier():
    """Simple Stochastic Gradient Descent Classifier with only two variables"""
    def __init__(self, alpha: float = 1.0, epochs: int = 10):
        self.epochs = epochs
        self.alpha = alpha
        self.epoch_losses: list[float] = []

        # Define the two variables to optimize
        b1 = torch.autograd.Variable(torch.tensor([0.01]), requires_grad=True)
        b2 = torch.autograd.Variable(torch.tensor([0.01]), requires_grad=True)
        self.weights = torch.Tensor([b1, b2]).reshape(2, 1)

    def fit(self, inputs, labels):
        
        for epoch in range(self.epochs):
            batch_losses = []

            for batch_x, batch_y in zip(inputs, labels):

                # Add a column of ones to x
                x_bias = torch.stack([torch.ones_like(batch_x), batch_x], dim=0)

                # Calculate p_x
                p_x = 1 / (1 + torch.pow(torch.e, - torch.matmul(self.weights.T, x_bias)))

                # Calculate the negative log likelihood loss
                loss = ((batch_y * torch.log(p_x)) + (1 - batch_y) * (1 - torch.log(1 - p_x))).sum()
                batch_losses.append(loss)

                # Calculate the gradient of the loss w.r.t. the inputs
                grad = (((p_x - batch_y) * x_bias).sum(dim=1) / len(batch_x)).reshape(2, 1)

                # Update the parameters b according to SGD formula
                self.weights = self.weights - self.alpha * grad

            self.epoch_losses.append(sum(batch_losses) / len(batch_losses))
            print(f"Epoch {epoch + 1} loss: {self.epoch_losses[-1]:.4f}")

    def predict(self, X):
        x_bias = torch.stack([torch.ones_like(X), X], dim=0)
        pred_prob = 1 / (1 + torch.pow(torch.e, -(torch.matmul(self.weights.T, x_bias)))).flatten()
        pred_y = (pred_prob > 0.5) * 1
        return pred_y, pred_prob

    def score(self, pred_y, label):
        metrics = precision_recall_fscore_support(pred_y, label, average='binary')
        return metrics
    


class BaseSGDClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class acting as the parent for all Stochastic Gradient Descent classifiers"""
    def __init__(
            self,
            loss: str = "hinge",
            penalty: str = 'l2',
            alpha: float = 0.001,
            learning_rate: str = "constant",
            eta0: int = 3,
            mini_batch_size: int = None,
            max_iter: int = 100,
            early_stop: bool = False,
            n_iter_no_change: int = 10,
            verbose: bool = False
        ) -> None:
        
        # Not implementing other strategies
        if loss not in ["hinge"]:
            raise NotImplementedError(f"Loss strategy {loss} not supported.")
        else:
            self.loss = loss

        if penalty not in ["l2"]:
            raise NotImplementedError(f"Regularization strategy {penalty} not supported.")
        else:
            self.penalty = penalty

        if learning_rate not in ["constant"]:
            raise NotImplementedError(f"Learning rate strategy {learning_rate} not supported.")
        else:
            self.learning_rate = learning_rate
        
        self.alpha = alpha
        self.eta0 = eta0
        self.mini_batch_size = mini_batch_size
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose

        # Init some properties
        self.losses: list = []
        self.coef_: np.ndarray = None
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        """Return predictions as an array of either -1, 0 or 1"""
        result = get_sign(self._soft_predict(X).ravel())
        return result

    def _soft_predict(self, X):
        """Return soft predictions, which is basically the dot product between X and coef"""
        X_mod = np.hstack((np.ones((len(X), 1)), X))
        result = X_mod.dot(self.coef_.transpose()).ravel()
        return result
    
    def score(self, X, y):
        """Use accuracy as the model performance score.
        accuracy = (number of pred == y) / (total number of y)
        """
        accuracy = sum((self.predict(X) == y).ravel()) / len(y)
        return accuracy
    
    def _calc_loss(self, y, soft_pred):
        """Return the regularized Hinge loss, which equals to regularization term + Hinge loss"""
        hinge_loss = sum([max([0, 1 - y[i] * soft_pred[i]]) for i in range(len(y))])
        regu = self.alpha / 2 * sum(self.coef_.ravel() ** 2)
        loss = regu + hinge_loss
        return loss
    
    def _calc_gradient(self, X, y):
        """Calculate the gradient using all data points"""
        pred = self._soft_predict(X)
        gradient = self.alpha * self.coef_ 
        
        for i, y_i in enumerate(y):
            if y_i * pred[i] < 1:
                gradient -= y_i * np.hstack((1, X[i]))
       
        return gradient


class SGDClassifier(BaseSGDClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y):
        """Fit parameters using gradient descent. 
        If mini_batch_size is supplied, use stochastic gradient descent.
        """
        self.coef_ = np.random.normal(size=(1, X.shape[1] + 1))
        best_loss = None
        no_improve = 0
        rng = np.random.default_rng()

        for i in range(self.max_iter):

            # Early stopping if current loss is greater than best_loss n times
            if best_loss is not None:
                if self.early_stop:
                    if loss > best_loss:
                        no_improve += 1

                    if no_improve == self.n_iter_no_change:
                        break
            
            # If mini_batch_size is specified, calculate loss and gradient using a randomly-sampled mini batch.
            if self.mini_batch_size is not None:
                batch_losses = []  # Loss list for all batches
                index_arr = np.array(range(len(X)))
                rng.shuffle(index_arr)
                batches = create_batches(index_arr, self.mini_batch_size)

                # Iterate through batches of training data, calculate loss and gradient, and update coef
                for batch in batches:
                    batch_loss = self._calc_loss(y[batch, ], self._soft_predict(X[batch,]))
                    gradient = self._calc_gradient(X[batch,], y[batch,])
                    
                    # Update coef at each batch
                    batch_losses.append(batch_loss)
                    self.coef_ = self.coef_ - self.eta0 * gradient
                
                loss = np.mean(batch_losses)  ## loss of an iteration is the average of all batch losses

            # If mini_batch_size is not specified, use full data
            else:
                loss = self._calc_loss(y, self._soft_predict(X))
                gradient = self._calc_gradient(X, y)
                self.coef_ = self.coef_ - self.eta0 * gradient

            # Save the iteration's loss
            # Save best loss and best coef to assign to the model at the end
            self.losses.append(loss)

            if not best_loss or loss < best_loss:
                best_loss = loss
                best_coef = self.coef_.copy()
            
            # Print training performance
            score = self.score(X, y)
            if self.verbose:
                print(f"Epoch {i + 1}, loss: {loss:.5f}, training accuracy: {score:.5f}", end="\r")

        if self.verbose:
            print()

        # Only retain the best coef
        self.coef_ = best_coef

        return self
    
    
class RMSPropSGDClassifier(BaseSGDClassifier):
    def __init__(self, gamma: float = 0.5, mini_batch_size: int = 1, *args, **kwargs) -> None:
        super().__init__(mini_batch_size=mini_batch_size, *args, **kwargs)
        self.gamma = gamma  # forget factor
        assert self.mini_batch_size is not None and self.mini_batch_size > 0 and isinstance(self.mini_batch_size, int), \
                "mini_batch_size must be an integer and at least 1"

    def fit(self, X, y):
        """Fit parameters using gradient descent. 
        If mini_batch_size is supplied, use stochastic gradient descent.
        """
        self.coef_ = np.random.normal(size=(1, X.shape[1] + 1))
        best_loss = None
        no_improve = 0
        rng = np.random.default_rng()

        for i in range(self.max_iter):
            
            # Early stopping if current loss is greater than best_loss n times
            if best_loss is not None:
                if self.early_stop:
                    if loss > best_loss:
                        no_improve += 1

                    if no_improve == self.n_iter_no_change:
                        break
            
            # Create batches of training data
            batch_losses = []  # Loss list for all batches
            index_arr = np.array(range(len(X)))
            rng.shuffle(index_arr)
            batches = create_batches(index_arr, self.mini_batch_size)

            # Initiate running average of squared gradients
            # If set init value of average = 0, in case derivative = 0, we have division by zero
            # If set init value of average = 1, in case gamma = 0.5 and derivative = 0, 
            # the second moving average will be (0.5 * 1) + (0.5 * 0^2) = 0.5, which makes sense
            running_avg = np.ones(self.coef_.shape)

            # Iterate through batches of training data, calculate loss and gradient, and update coef
            for batch in batches:
                batch_loss = self._calc_loss(y[batch, ], self._soft_predict(X[batch,]))
                gradient = self._calc_gradient(X[batch,], y[batch,])
                batch_losses.append(batch_loss)
                
                # RMSProp: Update coefficients using square root of a running mean square of the gradient
                # The running avg will help to even out the gradient of all batches
                running_avg = self.gamma * running_avg + (1 - self.gamma) * gradient ** 2
                self.coef_ = self.coef_ - (self.eta0 / np.sqrt(running_avg)) * gradient

            # Save best loss and best coef to assign to the model at the end
            loss = np.mean(batch_losses)  ## loss of an iteration is the average of all batch losses
            self.losses.append(loss)

            if not best_loss or loss < best_loss:
                best_loss = loss
                best_coef = self.coef_.copy()

            # Print training performance
            score = self.score(X, y)
            if self.verbose:
                print(f"Epoch {i + 1}, loss: {loss:.5f}, training accuracy: {score:.5f}", end="\r")

        if self.verbose:
            print()

        self.coef_ = best_coef
        return self
    