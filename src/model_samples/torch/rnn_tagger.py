# built-in packages
from pathlib import Path
from typing import Optional
from collections.abc import Iterable
from collections import defaultdict
from copy import deepcopy
import random

# torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# Others
from tqdm import tqdm
import numpy as np

# Others
from tqdm import tqdm


class PositionalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary: Optional[dict] = None, tagset: Optional[dict] = None) -> None:
        self.vocabulary: dict = vocabulary
        self.tagset: dict = tagset
        self.max_sentence_length_ = None

    def token_to_index(self, token):
        if token in self.vocabulary:
            return self.vocabulary[token]
        else:
            return self.vocabulary['<UNK>']

    def index_to_token(self, index):
        return list(self.vocabulary.keys())[index]

    def tag_to_index(self, tag):
        if tag in self.tagset:
            return self.tagset[tag]
        else:
            return self.tagset['<UNK>']

    def index_to_tag(self, index):
        return list(self.tagset.keys())[index]

    def fit(self, X: list[list], y: list[list]):
        """Learn token_vocab and POS tags from provided sentences and labels, produce """
        assert len(X) == len(y), "Sentences and labels have different length"
        assert np.all([len(sentence) == len(tags) for sentence, tags in zip(X, y)]), \
            "One of the sentences have mismatch word count - tag count"

        # Learn max sentence length so that we know how much to pad
        max_sentence_length = np.max([len(sentence) for sentence in X])
        self.max_sentence_length_ = max_sentence_length

        # Only learn the vocab when no pre-trained vocabs are provided
        if self.vocabulary is None and self.tagset is None:
            # Learn vocab
            token_set = {token for sentence in X for token in sentence}
            token_list = ['<UNK>'] + list(token_set) + ['<PAD>']
            self.vocabulary = {token: idx for idx, token in enumerate(token_list)}

            # Learn tags list
            tag_set = {tag for tags in y for tag in tags}
            tag_list = ['<UNK>'] + list(tag_set) + ['<PAD>']
            self.tagset = {tag: idx for idx, tag in enumerate(tag_list)}

        return self

    def transform(self, X: list[list], y: list[list]) -> tuple[torch.Tensor, torch.Tensor]:
        crow, col, token_val, tag_val = [], [], [], []

        # Iterate through sentences and labels, construct the necessary arrays to create CSR sparse matrices
        for i, (sentence, tags) in enumerate(zip(X, y)):
            crow.append(i * self.max_sentence_length_)

            for j, (token, tag) in enumerate(zip(sentence, tags)):
                col.append(j)

                # Append index of tokens and tags to the val arrays
                token_val.append(self.token_to_index(token))
                tag_val.append(self.tag_to_index(tag))

            # Add padding to the arrays:
            # Padding amount = difference between max length and the length of the current sentence
            # Columns to pad runs between (length of current sentence, max sentence length)
            padding_amt = self.max_sentence_length_ - len(sentence)
            col += list(range(len(sentence), self.max_sentence_length_))
            token_val += [self.token_to_index("<PAD>")] * padding_amt
            tag_val += [self.tag_to_index("<PAD>")] * padding_amt

        # Construct sparse matrices
        mat_size = (len(X), self.max_sentence_length_)
        tokens_sparse = torch.sparse_csr_tensor(crow, col, token_val, size=mat_size)
        tags_sparse = torch.sparse_csr_tensor(crow, col, tag_val, size=mat_size)

        return tokens_sparse.to_dense(), tags_sparse.to_dense()
    
    def fit_transform(self, X: list[list], y: list[list]) -> tuple[torch.Tensor, torch.Tensor]:
        self.fit(X, y)
        tokens, tags = self.transform(X, y)
        return tokens, tags


def evaluate_model(
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        pad_tag_idx: int
    ) -> float:
    """Evaluate the model on an inputs-targets set, using accuracy metric.

    Parameters
    ----------
    model : nn.Module
        Should be one of the two custom RNN taggers we defined.
    inputs : torch.Tensor
    targets : torch.Tensor
    pad_tag_idx : int
        Index of the <PAD> tag in the tagset to be ignored when calculating accuracy

    Returns
    -------
    float
        Accuracy metric (ignored the <PAD> tag)
    """

    # Make prediction
    scores = model(inputs)
    pred = scores.argmax(dim=2, keepdim=True).squeeze(dim=2)
                
    # Create a mask for ignoring <PAD> in the targets
    mask = targets != pad_tag_idx
    
    # Item pulls the value from the GPU automatically (if needed)
    correct = (pred[mask] == targets[mask]).sum().item()
    accuracy = correct / mask.sum().item()

    return accuracy


class BaseLSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, lstm_hidden_dim, vocabulary_size, tagset_size, padding_idx):
        """An LSTM based tagger

        word_embedding_dim
            The dimensionality of the word embedding
        lstm_hidden_dim
            The dimensionality of the hidden state in the LSTM
        vocabulary_size
            The number of unique tokens in the word embedding (including <PAD> etc)
        tagset_size
            The number of unique POS tags (not including <PAD>, as we don't want to predict it)
        """
        super(BaseLSTMTagger, self).__init__()                                          # We need to initialise the class we are inheriting from
        self.lstm_hidden_dim_ = lstm_hidden_dim                                     # This simply stores the parameters
        self.vocabulary_size_ = vocabulary_size
        self.tagset_size_ = tagset_size
        self.padding_idx = padding_idx

        self._get_word_embedding = nn.Embedding(num_embeddings=vocabulary_size,         # Creates the vector space for the input words
                                                embedding_dim=word_embedding_dim,
                                                padding_idx=padding_idx)
        self._lstm = nn.LSTM(input_size=word_embedding_dim,                         # The LSTM takes an embedded sentence as input, and outputs
                                hidden_size=lstm_hidden_dim,                           # vectors with dimensionality lstm_hidden_dim.
                                batch_first=True)
        self._fc = nn.Linear(lstm_hidden_dim, tagset_size)                          # The linear layer maps from the RNN output space to tag space
        self._softmax = nn.LogSoftmax(dim=1)                                        # Softmax of outputting PDFs over tags

        self.training_loss_ = list()                                                # For plotting
        self.training_accuracy_ = list()

        if torch.cuda.is_available():                                               # Move the model to the GPU (if we have one)
            self.cuda()

    def forward(self, X):
        """The forward pass through the network"""
        batch_size, max_sentence_length = X.size()

        embedded_sentences = self._get_word_embedding(X)                 # Sentences encoded as integers are mapped to vectors

        # Sentences need to be padded / truncated to a fixed size.
        # If set sentence length to be varied (no padding), this will cause an error. Explain:
        # In the original batch_iterator() function, x and y are padded,
        # then passed to the model. max_sentence_length is the (length of the longest sentence in the batch)
        # the X resulted from `pack_padded_sequence()` with variable-length sentences will 
        # create a tag_space matrix incompatible with fixed shape (64, 160, 18)
        # because the longest sentence in the batch may not be 160, but 100, for example.
        sentence_lengths = (X != self.padding_idx).sum(dim=1)        # Find the length of sentences
        sentence_lengths = sentence_lengths.long().cpu()                            # Ensure the correct format
        X = nn.utils.rnn.pack_padded_sequence(embedded_sentences, sentence_lengths, # Pack the embedded data
                                                batch_first=True, enforce_sorted=False)
        lstm_out, _ = self._lstm(X)
        X, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)         # Unpack the output from the LSTM
        X = X.contiguous().view(-1, X.shape[2])                                     # The output from the LSTM layer is flattened
        tag_space = self._fc(X)                                                     # Fully connected layer

        # Softmax is applied to normalise the outputs
        tag_scores = self._softmax(tag_space)  

        return tag_scores.view(batch_size, max_sentence_length, self.tagset_size_)


class TrainConfig():
    def __init__(
        self,
        optimizer_params: dict,
        loss_function: _Loss,
        n_epochs: int = 10,
        early_stop: bool = False,
        max_violations: int = 5
    ):
        self.optimizer_params = optimizer_params
        self.loss_function    = loss_function
        self.n_epochs         = n_epochs
        self.early_stop       = early_stop
        self.max_violations   = max_violations


class RNNTagger(nn.Module):
    def __init__(
        self, 
        encoder: PositionalEncoder,
        rnn_network: nn.Module = nn.LSTM,
        word_embedding_dim: int = 32,
        hidden_dim: int = 64,
        bidirectional: bool = False,
        dropout: float = 0.0,
        device: str = 'cpu'
    ):
        """An RNN based POS tagger, with some extensions

        Parameters
        ----------
        vocabulary_size : int
            The number of unique tokens in the word embedding (including <PAD> and <UNK>)
        tagset_size : int
            The number of unique POS tags (not including <PAD>, as we don't want to predict it)
        padding_idx : int
            Index of the <PAD> token in the vocabulary
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

        super(RNNTagger, self).__init__()
        self.hidden_dim_        = hidden_dim
        self.vocabulary_size_   = len(encoder.vocabulary)
        self.tagset_size_       = len(encoder.tagset)
        self.pad_token_idx_     = encoder.token_to_index('<PAD>')
        self.pad_tag_idx_       = encoder.tag_to_index('<PAD>')
        self.encoder_           = encoder
        
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device_ = 'cuda'
                self.cuda()
            else:
                self.device_ = 'cpu'
                print("CUDA not available. Run model on CPU.")
        else:
            self.device_ = 'cpu'

        # Initiate the word embedder. 
        # It is actually a nn.Linear module with a look up table to return the embedding 
        # corresponding to the token's positional index
        self._get_word_embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size_,
            embedding_dim=word_embedding_dim,
            padding_idx=self.pad_token_idx_
        )

        # Initiate the network 
        self._rnn_network = rnn_network(
            input_size=word_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        
        # Initiate a linear layer to transform output of _rnn_network to the tag space
        # Direction: 1 if uni-directional, 2 if bi-directional
        directions = bidirectional + 1
        self._fc = nn.Linear(hidden_dim * directions, self.tagset_size_)

        # Softmax layer to normalize the output of the linear layer 
        # to a probability distribution over the tag set
        self._softmax = nn.LogSoftmax(dim=1)

        # Store loss and accuracy to plot
        self.training_loss_ = list()
        self.training_accuracy_ = list()


    def forward(self, padded_sentences):
        """The forward pass through the network"""
        batch_size, max_sentence_length = padded_sentences.size()
        embedded_sentences = self._get_word_embedding(padded_sentences)
        print("embedded_sentence shape", embedded_sentences.data.shape)

        # Prepare a PackedSequence object, and pass data through the RNN
        sentence_lengths = (padded_sentences != self.pad_tag_idx_).sum(dim=1)
        sentence_lengths = sentence_lengths.long().cpu()

        packed_input = nn.utils.rnn.pack_padded_sequence(
            input=embedded_sentences, 
            lengths=sentence_lengths,
            batch_first=True, 
            enforce_sorted=False
        )
        rnn_output, _ = self._rnn_network(packed_input)  # Returned another PackedSequence
        print("packed_input shape: ", packed_input.data.shape)
        print("rnn_output shape: ", rnn_output.data.shape)
        
        # Unpack the PackedSequence
        unpacked_sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence=rnn_output, batch_first=True)
        unpacked_sequence = unpacked_sequence.contiguous().view(-1, unpacked_sequence.shape[2])
        print("unpacked sequence", unpacked_sequence.data.shape)

        # Pass data through the fully-connected linear layer
        tag_space = self._fc(unpacked_sequence)
        print("tag_space shape: ", tag_space.shape)

        # Softmax is applied to normalise the outputs
        tag_scores = self._softmax(tag_space)  

        return tag_scores.view(batch_size, max_sentence_length, self.tagset_size_)
  

    def fit(self, train_dataloader: DataLoader, train_config: TrainConfig, disable_progress_bar: bool = True) -> None:
        """Training loop for the RNN model. The loop will modify the model itself and returns nothing

        Parameters
        ----------
        train_dataloader : DataLoader
        train_config: TrainConfig
            An object containing various configs for the training loop
        train_encoder : PositionalEncoder
            The encoder providing the vocabulary and tagset for an internal batch_encoder
        Returns
        -------
        """
        # Make sure that the training process do not modify the initial model

        best_lost = float('inf')
        violations = 0
        loss_function = deepcopy(train_config.loss_function)
        optimizer = torch.optim.Adam(self.parameters(), **train_config.optimizer_params)

        for epoch in range(train_config.n_epochs):
            with tqdm(
                train_dataloader, 
                total   = len(train_dataloader), 
                unit    = "batch", 
                desc    = f"Epoch {epoch + 1}",
                disable = disable_progress_bar,
            ) as batches:
                
                for raw_inputs, raw_targets in batches:  
                    
                    # Initiate a batch-specific encoder that inherits the vocabulary from the pre-trained encoder
                    # to transform data in the batch.
                    batch_encoder = PositionalEncoder(vocabulary=self.encoder_.vocabulary, tagset=self.encoder_.tagset)

                    # max_sentence_length_ of each batch are allowed to be varied since it is learned here -> more memory-efficient
                    train_inputs, train_targets = batch_encoder.fit_transform(raw_inputs, raw_targets)
                    print("Input and target shape: ", train_inputs.shape, train_targets.shape)

                    # Move data to GPU if we want to and only if CUDA is available
                    if self.device_ == 'cuda':
                        if torch.cuda.is_available():
                            train_inputs = train_inputs.cuda()
                            train_targets = train_targets.cuda()
                        else:
                            print("CUDA not available. Run model on CPU.")

                    # Reset gradients, then run forward pass
                    self.zero_grad()
                    scores = self(train_inputs)

                    # Get loss value
                    loss = loss_function(scores.view(-1, self.tagset_size_), train_targets.view(-1))
                    
                    # Backward propagation. After each iteration through the batches, 
                    # accumulate the gradient for each theta
                    # Run the optimizer to update the parameters
                    loss.backward()
                    optimizer.step()

                    # Evaluate. Ignore <PAD> tag
                    accuracy = evaluate_model(self, train_inputs, train_targets, self.pad_tag_idx_)

                    # Save accuracy and loss for plotting
                    self.training_accuracy_.append(accuracy)
                    self.training_loss_.append(loss.item())

                    # Add loss and accuracy info to tqdm's progress bar
                    batches.set_postfix(loss=loss.item(), accuracy=accuracy) 

                    # Early stop:
                    if train_config.early_stop:
                        if loss < best_lost:
                            best_lost = loss
                            violations = 0
                        else:
                            violations += 1
                        
                        if violations == train_config.max_violations:
                            print(f"No improvement for {train_config.max_violations} epochs. Stop early.")
                            break

