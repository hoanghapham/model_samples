import torch
from sklearn.metrics import precision_recall_fscore_support


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
    
