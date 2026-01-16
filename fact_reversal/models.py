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


class SingleLayerQuarter(nn.Module):
    """Single hidden layer with quarter capacity (num_facts // 2 hidden units)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = max(1, vocab_size // 4)
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleLayerEighth(nn.Module):
    """Single hidden layer with 1/8 capacity."""

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = max(1, vocab_size // 8)
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleLayerSixteenth(nn.Module):
    """Single hidden layer with 1/16 capacity."""

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = max(1, vocab_size // 16)
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleLayerTiny(nn.Module):
    """Single hidden layer with 1/32 capacity."""

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = max(1, vocab_size // 32)
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class TiedWeights(nn.Module):
    """Autoencoder-style model with tied weights (W_out = W_in.T).

    This forces the model to use the same embedding space for inputs and outputs,
    which should help with learning bidirectional associations.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = vocab_size // 2
        # Only one weight matrix - output uses its transpose
        self.encoder = nn.Linear(vocab_size, hidden_size)
        # Separate bias for decoder
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tied weights.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Encode: x @ W.T + b (standard linear layer)
        hidden = torch.relu(self.encoder(x))
        # Decode: hidden @ W + decoder_bias (using transposed encoder weights)
        logits = nn.functional.linear(hidden, self.encoder.weight.T, self.decoder_bias)
        return logits


class EmbeddingDotProduct(nn.Module):
    """Embedding-based model where output is dot product with all embeddings.

    Each entity has a single embedding vector. The output logits are computed
    as the dot product between the input embedding and all embeddings.
    This naturally encourages symmetric relationships since the same embedding
    is used whether an entity appears as input or output.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        embedding_dim = vocab_size // 2
        # Embedding table: each row is an entity's embedding
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using embedding dot products.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Get input embeddings: (batch_size, vocab_size) @ (vocab_size, embed_dim)
        # For one-hot input, this is equivalent to looking up the embedding
        input_embed = x @ self.embedding  # (batch_size, embedding_dim)

        # Compute dot product with all embeddings
        # (batch_size, embedding_dim) @ (embedding_dim, vocab_size)
        logits = input_embed @ self.embedding.T  # (batch_size, vocab_size)

        return logits


class EmbeddingDotProductMasked(nn.Module):
    """Embedding dot product model with self-similarity masking.

    Same as EmbeddingDotProduct, but masks out the diagonal (self-predictions)
    by setting them to negative infinity. This prevents the model from
    trivially predicting the input itself.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        embedding_dim = vocab_size // 2
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked self-similarity.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        input_embed = x @ self.embedding  # (batch_size, embedding_dim)
        logits = input_embed @ self.embedding.T  # (batch_size, vocab_size)

        # Mask out self-predictions: set diagonal entries to -inf
        # For one-hot input x, x has a 1 at position i, so we mask position i
        # This is equivalent to: logits[batch_idx, input_idx] = -inf
        mask = x.bool()  # (batch_size, vocab_size)
        logits = logits.masked_fill(mask, float('-inf'))

        return logits


class EmbeddingCosine(nn.Module):
    """Embedding model using cosine similarity instead of dot product.

    Uses normalized embeddings so that similarity is based on angle rather
    than magnitude. This prevents any single embedding from dominating
    due to large norm.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        embedding_dim = vocab_size // 2
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)
        # Learnable temperature parameter for scaling logits
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cosine similarity.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Normalize embeddings to unit norm
        normalized_embed = nn.functional.normalize(self.embedding, dim=1)

        # Get input embedding (already unit norm after normalization)
        input_embed = x @ normalized_embed  # (batch_size, embedding_dim)

        # Cosine similarity with all embeddings
        # Since embeddings are normalized, dot product = cosine similarity
        cosine_sim = input_embed @ normalized_embed.T  # (batch_size, vocab_size)

        # Scale by temperature to sharpen/soften the distribution
        logits = cosine_sim * self.temperature

        return logits


class EmbeddingCosineMasked(nn.Module):
    """Cosine similarity model with self-similarity masking.

    Combines normalized embeddings (cosine similarity) with diagonal masking
    to prevent self-predictions.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        embedding_dim = vocab_size // 2
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cosine similarity with masking.

        Args:
            x: One-hot encoded input of shape (batch_size, vocab_size)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        normalized_embed = nn.functional.normalize(self.embedding, dim=1)
        input_embed = x @ normalized_embed
        cosine_sim = input_embed @ normalized_embed.T
        logits = cosine_sim * self.temperature

        # Mask self-predictions
        mask = x.bool()
        logits = logits.masked_fill(mask, float('-inf'))

        return logits
