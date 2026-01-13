"""Training and evaluation for the fact reversal experiment."""

import torch
import torch.nn as nn
from typing import NamedTuple
from fact_reversal.data import Dataset, DataSample


class TrainingMetrics(NamedTuple):
    """Metrics from training/evaluation."""
    loss: float
    accuracy: float
    forward_accuracy: float  # Accuracy on non-reversed samples
    reverse_accuracy: float  # Accuracy on reversed samples


def _samples_to_batch(samples: list[DataSample], vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert samples to one-hot encoded batch tensors.

    Args:
        samples: List of DataSample objects
        vocab_size: Size of vocabulary for one-hot encoding

    Returns:
        Tuple of (inputs, targets) as torch tensors
            - inputs: shape (batch_size, vocab_size), one-hot encoded
            - targets: shape (batch_size,), class indices
    """
    batch_size = len(samples)
    inputs = torch.zeros(batch_size, vocab_size)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(samples):
        inputs[i, sample.input_idx] = 1.0
        targets[i] = sample.output_idx

    return inputs, targets


def train_epoch(
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int = 32,
) -> TrainingMetrics:
    """Train for one epoch.

    Args:
        model: Neural network model
        dataset: Training dataset
        optimizer: Optimizer
        criterion: Loss function
        batch_size: Batch size for training

    Returns:
        TrainingMetrics from this epoch
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_forward_correct = 0
    total_reverse_correct = 0
    total_forward_samples = 0
    total_reverse_samples = 0

    # Process in batches
    for i in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[i : i + batch_size]

        inputs, targets = _samples_to_batch(batch_samples, dataset.vocab_size)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_samples)

        # Accuracy metrics
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).float()
        total_correct += correct.sum().item()

        # Track forward vs reverse accuracy
        for j, sample in enumerate(batch_samples):
            if sample.is_reverse:
                total_reverse_correct += correct[j].item()
                total_reverse_samples += 1
            else:
                total_forward_correct += correct[j].item()
                total_forward_samples += 1

    num_samples = len(dataset.samples)
    avg_loss = total_loss / num_samples
    avg_accuracy = total_correct / num_samples

    forward_acc = (
        total_forward_correct / total_forward_samples
        if total_forward_samples > 0
        else 0.0
    )
    reverse_acc = (
        total_reverse_correct / total_reverse_samples
        if total_reverse_samples > 0
        else 0.0
    )

    return TrainingMetrics(
        loss=avg_loss,
        accuracy=avg_accuracy,
        forward_accuracy=forward_acc,
        reverse_accuracy=reverse_acc,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    criterion: nn.Module,
    batch_size: int = 32,
) -> TrainingMetrics:
    """Evaluate model on a dataset.

    Args:
        model: Neural network model
        dataset: Dataset to evaluate on
        criterion: Loss function
        batch_size: Batch size for evaluation

    Returns:
        TrainingMetrics from evaluation
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_forward_correct = 0
    total_reverse_correct = 0
    total_forward_samples = 0
    total_reverse_samples = 0

    for i in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[i : i + batch_size]

        inputs, targets = _samples_to_batch(batch_samples, dataset.vocab_size)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * len(batch_samples)

        # Accuracy metrics
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).float()
        total_correct += correct.sum().item()

        # Track forward vs reverse accuracy
        for j, sample in enumerate(batch_samples):
            if sample.is_reverse:
                total_reverse_correct += correct[j].item()
                total_reverse_samples += 1
            else:
                total_forward_correct += correct[j].item()
                total_forward_samples += 1

    num_samples = len(dataset.samples)
    avg_loss = total_loss / num_samples
    avg_accuracy = total_correct / num_samples

    forward_acc = (
        total_forward_correct / total_forward_samples
        if total_forward_samples > 0
        else 0.0
    )
    reverse_acc = (
        total_reverse_correct / total_reverse_samples
        if total_reverse_samples > 0
        else 0.0
    )

    return TrainingMetrics(
        loss=avg_loss,
        accuracy=avg_accuracy,
        forward_accuracy=forward_acc,
        reverse_accuracy=reverse_acc,
    )


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
) -> dict:
    """Train a model and return results.

    Args:
        model: Neural network model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size

    Returns:
        Dictionary with training history and final results
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train": [],
        "val": [],
        "test": [],
    }

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_dataset, optimizer, criterion, batch_size
        )
        history["train"].append(train_metrics)

        # Validate
        if len(val_dataset.samples) > 0:
            val_metrics = evaluate(model, val_dataset, criterion, batch_size)
            history["val"].append(val_metrics)
        else:
            history["val"].append(
                TrainingMetrics(loss=0.0, accuracy=0.0, forward_accuracy=0.0, reverse_accuracy=0.0)
            )

        # Test (evaluate but don't optimize)
        test_metrics = evaluate(model, test_dataset, criterion, batch_size)
        history["test"].append(test_metrics)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Acc: {train_metrics.accuracy:.4f}, "
                f"Test Acc: {test_metrics.accuracy:.4f} "
                f"(Fwd: {test_metrics.forward_accuracy:.4f}, "
                f"Rev: {test_metrics.reverse_accuracy:.4f})"
            )

    return history
