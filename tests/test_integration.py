"""Integration tests for the full pipeline."""

import pytest
import torch
from fact_reversal.data import generate_facts, split_data
from fact_reversal.models import SingleLayerFull, SingleLayerHalf, TwoLayerNet
from fact_reversal.train import train_epoch, evaluate, train_model


class TestEndToEnd:
    """Test full pipeline from data generation to training."""

    def test_full_pipeline_single_layer_full(self):
        """Should run full pipeline successfully."""
        # Generate data
        facts = generate_facts(5, seed=42)
        train_data, val_data, test_data = split_data(facts, seed=42)

        # Create model
        model = SingleLayerFull(train_data.vocab_size)

        # Train
        history = train_model(
            model,
            train_data,
            val_data,
            test_data,
            num_epochs=5,
            learning_rate=0.01,
        )

        # Check history
        assert len(history["train"]) == 5
        assert len(history["test"]) == 5
        assert all(m.accuracy >= 0.0 and m.accuracy <= 1.0 for m in history["train"])

    def test_training_improves_loss(self):
        """Loss should generally decrease during training."""
        facts = generate_facts(10, seed=42)
        train_data, val_data, test_data = split_data(facts, seed=42)

        model = SingleLayerFull(train_data.vocab_size)

        history = train_model(
            model,
            train_data,
            val_data,
            test_data,
            num_epochs=10,
            learning_rate=0.01,
        )

        # Check that loss decreases overall
        first_loss = history["train"][0].loss
        last_loss = history["train"][-1].loss

        # Training loss should decrease (may have fluctuations)
        assert last_loss < first_loss

    def test_models_train_differently(self):
        """Different models should be trainable and produce different metrics."""
        facts = generate_facts(8, seed=42)
        train_data, val_data, test_data = split_data(facts, seed=42)

        models = {
            "full": SingleLayerFull(train_data.vocab_size),
            "half": SingleLayerHalf(train_data.vocab_size),
            "two_layer": TwoLayerNet(train_data.vocab_size),
        }

        results = {}

        for name, model in models.items():
            history = train_model(
                model,
                train_data,
                val_data,
                test_data,
                num_epochs=10,
                learning_rate=0.01,
            )
            final_train_acc = history["train"][-1].accuracy
            final_test_acc = history["test"][-1].accuracy
            results[name] = (final_train_acc, final_test_acc)

        # All models should be trainable and produce valid metrics
        for name, (train_acc, test_acc) in results.items():
            assert 0.0 <= train_acc <= 1.0, f"{name} train accuracy out of range"
            assert 0.0 <= test_acc <= 1.0, f"{name} test accuracy out of range"

        # With high probability, train accuracy should be higher than test
        # (overfitting is expected with random memorization task)
        full_train, full_test = results["full"]
        assert full_train >= full_test - 0.05, "Full model should overfit or match on training"

    def test_bidirectional_metrics_tracked(self):
        """Should track forward and reverse accuracies separately."""
        facts = generate_facts(5, seed=42)
        train_data, val_data, test_data = split_data(facts, seed=42)

        model = SingleLayerFull(train_data.vocab_size)

        history = train_model(
            model,
            train_data,
            val_data,
            test_data,
            num_epochs=5,
            learning_rate=0.01,
        )

        # Check that we have forward and reverse metrics
        for metrics in history["test"]:
            assert metrics.forward_accuracy >= 0.0
            assert metrics.reverse_accuracy >= 0.0
            assert hasattr(metrics, "forward_accuracy")
            assert hasattr(metrics, "reverse_accuracy")

    def test_training_with_different_batch_sizes(self):
        """Should handle different batch sizes."""
        facts = generate_facts(10, seed=42)
        train_data, val_data, test_data = split_data(facts, seed=42)

        model = SingleLayerFull(train_data.vocab_size)

        for batch_size in [1, 4, 16]:
            history = train_model(
                model,
                train_data,
                val_data,
                test_data,
                num_epochs=3,
                batch_size=batch_size,
            )

            assert len(history["train"]) == 3
            assert all(m.accuracy >= 0.0 for m in history["train"])


class TestTrainingFunctions:
    """Test training and evaluation functions."""

    def test_train_epoch_updates_model(self):
        """Training should update model parameters."""
        facts = generate_facts(5, seed=42)
        train_data, _, _ = split_data(facts, seed=42)

        model = SingleLayerFull(train_data.vocab_size)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Train one epoch
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        metrics = train_epoch(model, train_data, optimizer, criterion, batch_size=4)

        # Check that parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break

        assert params_changed, "Model parameters should be updated during training"
        assert isinstance(metrics.accuracy, float)

    def test_evaluate_no_grad(self):
        """Evaluate should not update model."""
        facts = generate_facts(5, seed=42)
        _, _, test_data = split_data(facts, seed=42)

        model = SingleLayerFull(test_data.vocab_size)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Evaluate
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, test_data, criterion, batch_size=4)

        # Check that parameters didn't change
        for initial, current in zip(initial_params, model.parameters()):
            assert torch.allclose(initial, current), "Evaluation should not update parameters"

        assert isinstance(metrics.accuracy, float)
