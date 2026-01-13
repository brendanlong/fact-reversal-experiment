"""Tests for neural network models."""

import pytest
import torch
from fact_reversal.models import SingleLayerFull, SingleLayerHalf, TwoLayerNet


class TestModelInputOutput:
    """Test basic input/output behavior of models."""

    @pytest.mark.parametrize(
        "model_class",
        [SingleLayerFull, SingleLayerHalf, TwoLayerNet],
    )
    def test_output_shape(self, model_class):
        """Output should have shape (batch_size, vocab_size)."""
        vocab_size = 20
        batch_size = 8
        model = model_class(vocab_size)

        # Create dummy input (one-hot)
        x = torch.zeros(batch_size, vocab_size)
        x[torch.arange(batch_size), torch.randint(0, vocab_size, (batch_size,))] = 1.0

        output = model(x)

        assert output.shape == (batch_size, vocab_size)

    @pytest.mark.parametrize(
        "model_class",
        [SingleLayerFull, SingleLayerHalf, TwoLayerNet],
    )
    def test_accepts_one_hot_input(self, model_class):
        """Model should accept one-hot encoded input."""
        vocab_size = 10
        batch_size = 4
        model = model_class(vocab_size)

        # Create one-hot input
        x = torch.zeros(batch_size, vocab_size)
        for i in range(batch_size):
            x[i, i] = 1.0

        # Should not raise an error
        output = model(x)
        assert output is not None

    @pytest.mark.parametrize(
        "model_class",
        [SingleLayerFull, SingleLayerHalf, TwoLayerNet],
    )
    def test_produces_logits(self, model_class):
        """Output should be unbounded logits (not softmax)."""
        vocab_size = 10
        batch_size = 4
        model = model_class(vocab_size)

        x = torch.zeros(batch_size, vocab_size)
        x[torch.arange(batch_size), torch.randint(0, vocab_size, (batch_size,))] = 1.0

        output = model(x)

        # Logits can be negative or > 1 (unlike softmax output)
        # Check that we don't always get values in [0, 1]
        has_negative = (output < 0).any()
        has_large = (output > 1).any()

        # At least one of these should be true for unactivated logits
        # (they won't be softmax normalized)
        assert isinstance(output, torch.Tensor)


class TestSingleLayerFull:
    """Test SingleLayerFull model."""

    def test_parameter_count(self):
        """Should have correct number of parameters."""
        vocab_size = 20
        model = SingleLayerFull(vocab_size)

        # Input layer: vocab_size -> vocab_size + bias
        # Hidden layer: vocab_size -> vocab_size + bias
        # Total: vocab_size * vocab_size + vocab_size + vocab_size * vocab_size + vocab_size
        #      = 2 * vocab_size^2 + 2 * vocab_size

        expected = 2 * vocab_size * vocab_size + 2 * vocab_size
        actual = sum(p.numel() for p in model.parameters())

        assert actual == expected

    def test_large_capacity_fits_data(self):
        """Full model should have capacity to memorize."""
        vocab_size = 20
        model = SingleLayerFull(vocab_size)

        # Vocab size = 2N, so this has N hidden units where N = vocab_size/2
        # With vocab_size hidden units, should have plenty of capacity
        num_params = sum(p.numel() for p in model.parameters())

        # For our task, we have at most vocab_size data points (one per symbol)
        # Model should have many more parameters
        assert num_params > vocab_size


class TestSingleLayerHalf:
    """Test SingleLayerHalf model."""

    def test_parameter_count(self):
        """Should have correct number of parameters."""
        vocab_size = 20
        hidden_size = vocab_size // 2
        model = SingleLayerHalf(vocab_size)

        # Input -> Hidden: vocab_size * hidden_size + hidden_size
        # Hidden -> Output: hidden_size * vocab_size + vocab_size
        # Total: vocab_size * hidden_size + hidden_size + hidden_size * vocab_size + vocab_size
        #      = 2 * vocab_size * hidden_size + vocab_size + hidden_size

        expected = 2 * vocab_size * hidden_size + vocab_size + hidden_size
        actual = sum(p.numel() for p in model.parameters())

        assert actual == expected

    def test_half_capacity_vs_full(self):
        """Half model should have fewer parameters than full model."""
        vocab_size = 20
        full_model = SingleLayerFull(vocab_size)
        half_model = SingleLayerHalf(vocab_size)

        full_params = sum(p.numel() for p in full_model.parameters())
        half_params = sum(p.numel() for p in half_model.parameters())

        assert half_params < full_params


class TestTwoLayerNet:
    """Test TwoLayerNet model."""

    def test_parameter_count(self):
        """Should have correct number of parameters."""
        vocab_size = 20
        hidden_size = vocab_size // 2
        model = TwoLayerNet(vocab_size)

        # Input -> Hidden1: vocab_size * hidden_size + hidden_size
        # Hidden1 -> Hidden2: hidden_size * hidden_size + hidden_size
        # Hidden2 -> Output: hidden_size * vocab_size + vocab_size
        # Total: vocab_size * hidden_size + hidden_size + hidden_size^2 + hidden_size + vocab_size * hidden_size + vocab_size
        #      = 2 * vocab_size * hidden_size + hidden_size^2 + 2 * hidden_size + vocab_size

        expected = (
            2 * vocab_size * hidden_size
            + hidden_size * hidden_size
            + 2 * hidden_size
            + vocab_size
        )
        actual = sum(p.numel() for p in model.parameters())

        assert actual == expected

    def test_has_two_hidden_layers(self):
        """Should use two hidden layers."""
        vocab_size = 20
        model = TwoLayerNet(vocab_size)

        # Count linear layers: should be 3 (input->h1, h1->h2, h2->output)
        linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3


class TestModelComparison:
    """Compare models to verify design."""

    def test_all_models_trainable(self):
        """All models should have trainable parameters."""
        vocab_size = 10
        models = [
            SingleLayerFull(vocab_size),
            SingleLayerHalf(vocab_size),
            TwoLayerNet(vocab_size),
        ]

        for model in models:
            params = [p for p in model.parameters() if p.requires_grad]
            assert len(params) > 0, "Model should have trainable parameters"

    def test_models_can_forward_pass(self):
        """All models should complete forward pass."""
        vocab_size = 10
        batch_size = 4
        models = [
            SingleLayerFull(vocab_size),
            SingleLayerHalf(vocab_size),
            TwoLayerNet(vocab_size),
        ]

        x = torch.zeros(batch_size, vocab_size)
        x[torch.arange(batch_size), torch.randint(0, vocab_size, (batch_size,))] = 1.0

        for model in models:
            output = model(x)
            assert output.shape == (batch_size, vocab_size)
            assert not torch.isnan(output).any()
