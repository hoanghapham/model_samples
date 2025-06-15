from typing import Optional, Callable
from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from model_samples.utils import TrainConfig


class PositionalEncoder(BaseEstimator, TransformerMixin):
    """Positional encoder to encode text into positional vectors. Used for RNN models.
    """
    def __init__(
            self, 
            vocabulary: Optional[Iterable] = None,
            tokenizer: Optional[Callable] = None,
            disable_progress_bar: bool = True,
        ) -> None:
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer or self.build_tokenizer()
        self.max_sentence_length_ = 0
        self.disable_progress_bar = disable_progress_bar

    def token_to_index(self, token):
        if token in self.vocabulary:
            return self.vocabulary[token]
        else:
            return self.vocabulary['<UNK>']

    def index_to_token(self, index):
        return list(self.vocabulary.keys())[index]

    def fit(self, X: list[str], y: Optional[list] = None):
        """Learn vocabulary from provided sentences and labels
        """
        vocabulary: set = set()
        max_sentence_length = 0

        # Iterate through sentences in the input, tokenize, and update vocabulary
        for sentence in tqdm(X, total=len(X), desc="Fit\t\t", unit="sample", disable=self.disable_progress_bar):
            tokens = self.tokenizer(sentence)
            if len(tokens) > max_sentence_length:
                max_sentence_length = len(tokens)

            if self.vocabulary is None:
                for token in tokens:
                    vocabulary.add(token)
        
        if self.vocabulary is None:
            vocab_list = ['<UNK>'] + list(vocabulary) + ['<PAD>']
            self.vocabulary = {word: idx for idx, word in enumerate(vocab_list)}

        self.max_sentence_length_ = max_sentence_length

        return self

    def transform(self, X, y: Optional[list] = None):

        crow, col, token_val = [], [], []

        # Iterate through sentences, construct the necessary arrays to create CSR sparse matrix
        for i, text in tqdm(enumerate(X), total=len(X), desc="Transform\t", unit="sample", disable=self.disable_progress_bar):
            tokens = self.tokenizer(text)
            crow.append(i * self.max_sentence_length_)

            for j, token in enumerate(tokens):
                col.append(j)

                # Append index of tokens and tags to the val arrays
                if token in self.vocabulary:
                    token_val.append(self.token_to_index(token))
                else:
                    token_val.append(self.token_to_index("<UNK>"))

            # Add padding to make all sentences have the same length
            padding_amt = self.max_sentence_length_ - len(tokens)
            token_val += [self.token_to_index("<PAD>")] * padding_amt

            # Column index of the paddings runs from (len of tokens) to max_sentence_length
            col += list(range(len(tokens), self.max_sentence_length_))

        assert len(token_val) == self.max_sentence_length_ * len(X), \
            f"Length of token_val is incorrect: {len(token_val)} != {self.max_sentence_length_ * len(X)}"

        # Construct sparse matrices
        mat_size = (len(X), self.max_sentence_length_)
        tokens_sparse = torch.sparse_csr_tensor(crow, col, token_val, size=mat_size, dtype=torch.long)

        return tokens_sparse.to_dense()
    
    def build_tokenizer(self):
        def simple_tokenizer(sentence):
            words: list = sentence.split(' ')
            new_words = []

            for word in words:
                # If there is no punctuation in the word, add to the final list
                # Else, split the words further at the punctuations
                if all([char.isalnum() for char in word]):
                    new_words.append(word)

                else:
                    tmp = ''
                    # Iterate through characters. When encounter a punctuation,
                    # add the previous characters as a word, then add the punctuation
                    for char_idx, char in enumerate(word):
                        if char.isalnum():
                            tmp += char
                            if char_idx == len(word) - 1:
                                new_words.append(tmp)
                        else:
                            if char_idx > 0:
                                new_words.append(tmp)
                            new_words.append(char)
                            tmp = ''
            return new_words
        return simple_tokenizer


class RNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: PositionalEncoder,
        rnn_network: nn.LSTM | nn.GRU = nn.LSTM,
        word_embedding_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 1,  # binary classification only need 1 output
        bidirectional: bool = False,
        dropout: float = 0.0,
        device: str = 'cpu'
    ):
        """An RNN based classifier

        Parameters
        ----------
        encoder : PositionalEncoder
        rnn_network : nn.Module, optional
            The network type to be used, can be either nn.LSTM or nn.GRU. By default nn.LSTM
        word_embedding_dim : int, optional
            The dimensionality of the word embedding, by default 32
        hidden_dim : int, optional
            The dimensionality of the hidden state in the RNN, by default 64
        bidirectional : bool, optional
            Specify if the RNN is bi-directional or not, by default False
        dropout : float, optional
            Ratio of random weight drop-out while training, by default 0.0
        device : str, optional
            Device to train the model on, can be either 'cuda' or 'cpu'. By default 'cpu'
        """

        assert rnn_network in [nn.LSTM, nn.GRU], "rnn_network must be nn.LSTM or nn.GRU"
        assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"

        super().__init__()
        self.hidden_dim_        = hidden_dim
        self.vocabulary_size_   = len(encoder.vocabulary)
        self.output_dim_        = output_dim
        self.pad_token_idx_     = encoder.token_to_index('<PAD>')
        self.encoder_           = encoder

        if device == 'cuda':
            if torch.cuda.is_available():
                self.to('cuda')
                self.device = 'cuda'
            else:
                print("CUDA not available. Run model on CPU.")
                self.device = 'cpu'
                self.to('cpu')
        else:
            self.device = 'cpu'

        # Initiate the word embedder.
        # It is actually a nn.Linear module with a look up table to return the embedding
        # corresponding to the token's positional index
        self._get_word_embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size_,
            embedding_dim=word_embedding_dim,
            padding_idx=self.pad_token_idx_
        ).to(self.device)

        # Initiate the network
        self._rnn_network = rnn_network(
            input_size=word_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        ).to(self.device)

        # Initiate a linear layer to transform output of _rnn_network to the class space
        # Direction: 1 if uni-directional, 2 if bi-directional
        # This is a binary classification, so only need 1 output unit
        directions = bidirectional + 1
        self._fc = nn.Linear(hidden_dim * directions, 1).to(self.device)

        # Sigmoid to convert output to probability between 0 and 1
        self._sigmoid = nn.Sigmoid()

        # Store loss and accuracy to plot
        self.training_loss_ = list()
        self.training_accuracy_ = list()


    def forward(self, padded_sentences: torch.Tensor) -> torch.Tensor:
        """The forward pass through the network"""
        batch_size, max_sentence_length = padded_sentences.size()
        embedded_sentences = self._get_word_embedding(padded_sentences)

        # Prepare a PackedSequence object, and pass data through the RNN
        sentence_lengths = (padded_sentences != self.pad_token_idx_).sum(dim=1)
        sentence_lengths = sentence_lengths.long().cpu()

        packed_input = nn.utils.rnn.pack_padded_sequence(
            input=embedded_sentences,
            lengths=sentence_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        rnn_output, _ = self._rnn_network(packed_input)  # Returned another PackedSequence

        # Unpack the PackedSequence
        unpacked_sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence=rnn_output, batch_first=True)
        unpacked_sequence = unpacked_sequence.contiguous().view(-1, unpacked_sequence.shape[2])

        # Pass data through the fully-connected linear layer
        class_space = self._fc(unpacked_sequence)

        # Reshape data Example size: (64, 2000, 1)
        reshaped = class_space.view(batch_size, max_sentence_length, 1)

        # With RNN, need to collapse the soft prediction (logit) into a one-dimension vector
        # Get the last token in each sentence as the output of RNN
        collapsed_output = torch.stack([reshaped[i, j-1] for i, j in enumerate(sentence_lengths)]).squeeze().to(self.device)

        return collapsed_output
    

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions for the input tensor"""
        logits = self.forward(x.to(self.device))
        pred = (self._sigmoid(logits) >= 0.5).squeeze() * 1.0  # Convert to 1-0
        return pred

    def fit(
        self, 
        train_dataloader: DataLoader,
        train_config: TrainConfig,
        disable_progress_bar: bool = True
    ) -> None:
        """Training loop for the RNN model. The loop will modify the model itself and returns nothing

        Parameters
        ----------
        train_dataloader : DataLoader
        train_config: TrainConfig
            An object containing various configs for the training loop
        train_encoder : PositionalEncoder
            The encoder providing the vocabulary and tagset for an internal batch_encoder
        """

        best_lost = float('inf')
        violations = 0
        optimizer = torch.optim.Adam(self.parameters(), **train_config.optimizer_params)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(train_config.num_epochs):
            with tqdm(
                train_dataloader,
                total   = len(train_dataloader),
                unit    = "batch",
                desc    = f"Epoch {epoch + 1}",
                disable = disable_progress_bar,
            ) as batches:

                for ids, speakers, raw_inputs, raw_targets in batches:

                    # Initiate a batch-specific encoder that inherits the vocabulary from the pre-trained encoder
                    # to transform data in the batch.
                    batch_encoder = PositionalEncoder(vocabulary=self.encoder_.vocabulary,)

                    # max_sentence_length_ of each batch are allowed to be varied since it is learned here -> more memory-efficient
                    #
                    train_inputs = batch_encoder.fit_transform(raw_inputs).to(self.device)
                    train_targets = torch.as_tensor(raw_targets, dtype=torch.float).to(self.device)  # nn.CrossEntropyLoss() require target to be float

                    # Reset gradients, then run forward pass
                    self.zero_grad()
                    logits = self(train_inputs)

                    # Calc loss
                    loss = loss_function(logits.view(-1), train_targets.view(-1))

                    # Backward propagation. After each iteration through the batches,
                    # accumulate the gradient for each theta
                    # Run the optimizer to update the parameters
                    loss.backward()
                    optimizer.step()

                    # Evaluate with training batch accuracy
                    pred = self.predict(train_inputs)
                    correct = (pred == train_targets).sum().item()
                    accuracy = correct / len(train_targets)

                    # Save accuracy and loss for plotting
                    self.training_accuracy_.append(accuracy)
                    self.training_loss_.append(loss.item())

                    # Add loss and accuracy info to tqdm's progress bar
                    batches.set_postfix(loss=loss.item(), batch_accuracy=accuracy)

                    # Early stop:
                    if train_config.early_stop:
                        if loss < best_lost:
                            best_lost = loss
                            violations = 0
                        else:
                            violations += 1

                        if violations == train_config.violation_limit:
                            print(f"No improvement for {train_config.violation_limit} epochs. Stop early.")
                            break