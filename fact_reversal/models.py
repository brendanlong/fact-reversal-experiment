"""Neural network models for the fact reversal experiment."""

import torch
import torch.nn as nn


class SingleLayerFull(nn.Module):
    """Single hidden layer with full capacity (2*num_facts hidden units)."""

    def __init__(self, vocab_size: int):
        """
        Args:
            vocab_size: Total vocabulary size (2 * num_facts)
        """
        super().__init__()
        hidden_size = vocab_size  # Full capacity to memorize
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        return self.net(x)


class SingleLayerHalf(nn.Module):
    """Single hidden layer with half capacity (num_facts hidden units)."""

    def __init__(self, vocab_size: int):
        """
        Args:
            vocab_size: Total vocabulary size (2 * num_facts)
        """
        super().__init__()
        hidden_size = vocab_size // 2  # Half capacity
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        return self.net(x)


class TwoLayerNet(nn.Module):
    """Two hidden layers with num_facts units each."""

    def __init__(self, vocab_size: int):
        """
        Args:
            vocab_size: Total vocabulary size (2 * num_facts)
        """
        super().__init__()
        hidden_size = vocab_size // 2  # num_facts units
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        return self.net(x)
