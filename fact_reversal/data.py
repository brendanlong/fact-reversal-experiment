"""Data generation and splitting for the fact reversal experiment."""

import numpy as np
from typing import NamedTuple


class FactPair(NamedTuple):
    """A bidirectional fact pair."""
    a: int
    b: int


class DataSample(NamedTuple):
    """A single training sample."""
    input_idx: int  # The input number (will be one-hot encoded)
    output_idx: int  # The output number to predict
    is_reverse: bool  # Whether this is the reverse direction of the fact


class Dataset(NamedTuple):
    """A dataset of samples."""
    samples: list[DataSample]
    vocab_size: int  # Total number of unique values (2*num_facts)


def generate_facts(num_facts: int, seed: int = None) -> list[FactPair]:
    """Generate N random bidirectional facts.

    Creates a list of numbers 0 to 2N-1, shuffles it, and pairs the two halves.

    Args:
        num_facts: Number of facts to generate
        seed: Random seed for reproducibility

    Returns:
        List of FactPair objects
    """
    if seed is not None:
        np.random.seed(seed)

    vocab_size = 2 * num_facts
    numbers = np.arange(vocab_size)
    np.random.shuffle(numbers)

    # Split in half and create pairs
    first_half = numbers[:num_facts]
    second_half = numbers[num_facts:]

    facts = [FactPair(a=first_half[i], b=second_half[i]) for i in range(num_facts)]
    return facts


def split_data(
    facts: list[FactPair],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split facts into train/val/test sets.

    Ensures that for each fact, at least one direction appears in the training set.

    Args:
        facts: List of FactPair objects
        val_ratio: Fraction of samples for validation (of non-training samples)
        test_ratio: Fraction of samples for test (of non-training samples)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if seed is not None:
        np.random.seed(seed)

    vocab_size = 2 * len(facts)

    # Create samples: for each fact, we can have forward and reverse directions
    # We'll decide which directions go to train/val/test
    all_samples = []

    for fact_idx, fact in enumerate(facts):
        # Forward direction: input a -> output b
        all_samples.append(DataSample(
            input_idx=fact.a,
            output_idx=fact.b,
            is_reverse=False,
        ))
        # Reverse direction: input b -> output a
        all_samples.append(DataSample(
            input_idx=fact.b,
            output_idx=fact.a,
            is_reverse=True,
        ))

    # Now split: ensure each fact has at least one direction in training
    # We'll go through each fact and randomly assign one direction to training
    train_samples = []
    remaining_samples = []

    for fact_idx, fact in enumerate(facts):
        forward = all_samples[2 * fact_idx]
        reverse = all_samples[2 * fact_idx + 1]

        # Randomly choose which direction goes to training (or both)
        # To ensure at least one: randomly pick 1 or 2 directions for training
        num_to_train = np.random.choice([1, 2])

        if num_to_train == 1:
            # Pick one randomly for training
            if np.random.rand() < 0.5:
                train_samples.append(forward)
                remaining_samples.append(reverse)
            else:
                train_samples.append(reverse)
                remaining_samples.append(forward)
        else:
            # Both go to training
            train_samples.append(forward)
            train_samples.append(reverse)

    # Split remaining samples into validation and test
    num_remaining = len(remaining_samples)
    num_val = max(1, int(num_remaining * val_ratio / (val_ratio + test_ratio)))

    shuffled_indices = np.random.permutation(num_remaining)
    val_indices = shuffled_indices[:num_val]
    test_indices = shuffled_indices[num_val:]

    val_samples = [remaining_samples[i] for i in val_indices]
    test_samples = [remaining_samples[i] for i in test_indices]

    # Shuffle training samples (using indices to preserve namedtuples)
    train_indices = np.random.permutation(len(train_samples))
    train_samples_shuffled = [train_samples[i] for i in train_indices]

    return (
        Dataset(samples=train_samples_shuffled, vocab_size=vocab_size),
        Dataset(samples=val_samples, vocab_size=vocab_size),
        Dataset(samples=test_samples, vocab_size=vocab_size),
    )
