from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_samples.utils import TrainConfig

class NNClassifier(nn.Module):
    """Simple neural-network based classifier built using torch's modules
    """
    def __init__(
            self, 
            input_size, 
            hidden_size = 64, 
            n_linear_layers: int = 1,
            output_size = 1,  # binary classification only need 1 output
            positive_pred_threshold: float = 0.5,
            pos_weight: float = 1.0,
            dropout: float = 0,
            device = 'cpu',
        ):
        assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"
        super().__init__()

        # Define the layers of the neural network

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Hidden layers can be dynamically generated
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout), nn.ReLU()) 
            for i in range(n_linear_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)

        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.cuda()
            else:
                print("CUDA not available. Run model on CPU")
                self.device = 'cpu'
                self.cpu()
        else:
            self.device = 'cpu'
            self.cpu()

        self.sigmoid = nn.Sigmoid()
        self.positive_pred_threshold = positive_pred_threshold
        self.pos_weight = torch.Tensor([pos_weight]).to(self.device)

        self.training_accuracy_: list = []
        self.training_loss_: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network. Return the logits

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        # Define the forward pass of the neural network
        activation = self.input_layer(x.to(self.device))

        for layer in self.hidden_layers:
            activation = layer(activation)
        
        logits = self.output_layer(activation).squeeze()

        return logits

    def predict(self, x: torch.Tensor, positive_pred_threshold: float = None) -> torch.Tensor:
        """Output prediction labels (1, 0)

        Parameters
        ----------
        x : torch.Tensor
        positive_pred_threshold : float, optional
            If the result of sigmoid(logits) is greater than the threshold, then the prediction is 1.

        Returns
        -------
        torch.Tensor
            A tensor of labels (0, 1)
        """
        if not positive_pred_threshold:
            positive_pred_threshold = self.positive_pred_threshold
        logits = self.forward(x.to(self.device))
        pred = (self.sigmoid(logits) >= positive_pred_threshold).squeeze() * 1.0  # Convert to 1-0 labels
        return pred

    def fit(
        self,
        train_dataloader: DataLoader,
        train_config: TrainConfig,
        disable_progress_bar: bool = True
    ) -> None:
        """Train the model

        Parameters
        ----------
        train_dataloader : DataLoader
        train_config : TrainConfig
        disable_progress_bar : bool, optional
            If True, disable the progress bar, by default True
        """
        best_loss = float('inf')
        violations = 0
        loss_function = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adjust learning rate
        print()

        for epoch in range(train_config.num_epochs):
            self.train()
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch", disable=disable_progress_bar) as train_batches:
                for X_train, y_train in train_batches:
                    X_train = X_train.to(self.device)
                    y_train = y_train.float().to(self.device)
                    optimizer.zero_grad()
                    logits = self(X_train)

                    # print(logits.shape, y_train.shape)
                    loss = loss_function(logits, y_train)
                    loss.backward()
                    optimizer.step()

                    # Evaluate on train set 
                    self.eval()
                    pred = self.predict(X_train)
                    corrects = (pred == y_train).sum().item()
                    accuracy = corrects / len(y_train)

                    self.training_accuracy_.append(accuracy)
                    self.training_loss_.append(loss.item())

                    train_batches.set_postfix(batch_accuracy=accuracy, loss=loss.item())

                    torch.cuda.empty_cache()
            
                    if loss < best_loss:
                        best_loss = loss
                        violations = 0
                    else:
                        violations += 1

                    if train_config.early_stop:
                        if violations >= train_config.violation_limit:
                            print(f"Validation loss did not improve for {train_config.violation_limit} iterations. Stopping early.")
                            break
