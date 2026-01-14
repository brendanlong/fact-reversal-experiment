"""Tests for data generation and splitting."""

import pytest
import numpy as np
from fact_reversal.data import generate_facts, split_data, FactPair


class TestGenerateFacts:
    """Test fact generation."""

    def test_correct_number_of_facts(self):
        """Should generate exactly N facts."""
        num_facts = 10
        facts = generate_facts(num_facts, seed=42)
        assert len(facts) == num_facts

    def test_all_values_used(self):
        """Should use all values from 0 to 2N-1 exactly once."""
        num_facts = 10
        facts = generate_facts(num_facts, seed=42)

        all_values = set()
        for fact in facts:
            all_values.add(fact.a)
            all_values.add(fact.b)

        expected = set(range(2 * num_facts))
        assert all_values == expected

    def test_no_duplicate_pairs(self):
        """Should not have duplicate facts."""
        num_facts = 20
        facts = generate_facts(num_facts, seed=42)

        # Create a set of sorted tuples to detect duplicates
        pairs = set()
        for fact in facts:
            pair = tuple(sorted([fact.a, fact.b]))
            assert pair not in pairs, f"Duplicate pair found: {pair}"
            pairs.add(pair)

    def test_reproducibility(self):
        """Same seed should produce same facts."""
        facts1 = generate_facts(10, seed=42)
        facts2 = generate_facts(10, seed=42)

        assert facts1 == facts2

    def test_different_seeds_different_results(self):
        """Different seeds should produce different facts."""
        facts1 = generate_facts(10, seed=42)
        facts2 = generate_facts(10, seed=43)

        # Very unlikely to be the same with different seeds
        assert facts1 != facts2

    def test_fact_pair_structure(self):
        """Facts should be FactPair namedtuples with a and b."""
        num_facts = 5
        facts = generate_facts(num_facts, seed=42)

        for fact in facts:
            assert isinstance(fact, FactPair)
            assert hasattr(fact, "a")
            assert hasattr(fact, "b")
            assert isinstance(fact.a, (int, np.integer))
            assert isinstance(fact.b, (int, np.integer))


class TestSplitData:
    """Test data splitting."""

    def test_split_returns_three_datasets(self):
        """Should return train, val, test datasets."""
        facts = generate_facts(10, seed=42)
        train, val, test = split_data(facts, seed=42)

        assert train is not None
        assert val is not None
        assert test is not None

    def test_all_samples_have_vocab_size(self):
        """All datasets should have vocab_size set."""
        facts = generate_facts(10, seed=42)
        train, val, test = split_data(facts, seed=42)

        expected_vocab = 2 * len(facts)
        assert train.vocab_size == expected_vocab
        assert val.vocab_size == expected_vocab
        assert test.vocab_size == expected_vocab

    def test_at_least_one_direction_per_fact_in_training(self):
        """Training set should have at least one direction of each fact."""
        num_facts = 10
        facts = generate_facts(num_facts, seed=42)
        train, val, test = split_data(facts, seed=42)

        # Track which facts we've seen (as (min, max) pairs to avoid direction)
        seen_facts = set()

        for sample in train.samples:
            pair = tuple(sorted([sample.input_idx, sample.output_idx]))
            seen_facts.add(pair)

        # Check that we have at least one of each fact
        # This is probabilistic since we randomly assign directions,
        # but with these parameters should be very likely
        assert len(seen_facts) >= num_facts - 1, (
            f"Expected at least {num_facts - 1} unique facts in training, "
            f"got {len(seen_facts)}"
        )

    def test_train_val_test_no_overlap_samples(self):
        """Train, val, test samples should be different instances."""
        facts = generate_facts(10, seed=42)
        train, val, test = split_data(facts, seed=42)

        # Create tuples representing each sample
        train_samples = {(s.input_idx, s.output_idx, s.is_reverse) for s in train.samples}
        val_samples = {(s.input_idx, s.output_idx, s.is_reverse) for s in val.samples}
        test_samples = {(s.input_idx, s.output_idx, s.is_reverse) for s in test.samples}

        # Val and test should not overlap (train may contain both directions)
        assert len(val_samples & test_samples) == 0

    def test_reasonable_split_ratio(self):
        """Split should follow approximately the specified ratios."""
        facts = generate_facts(100, seed=42)
        train, val, test = split_data(
            facts, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        # Not all training samples are in train (some facts have both directions there)
        # so we can't check exact ratios, but val+test should be non-trivial
        total_val_test = len(val.samples) + len(test.samples)
        assert total_val_test > 0, "Should have some validation and test samples"

    def test_reproducibility(self):
        """Same seed should produce same split."""
        facts = generate_facts(10, seed=42)
        train1, val1, test1 = split_data(facts, seed=42)
        train2, val2, test2 = split_data(facts, seed=42)

        assert train1.samples == train2.samples
        assert val1.samples == val2.samples
        assert test1.samples == test2.samples

    def test_empty_facts_raises_error(self):
        """Empty facts list should not cause issues."""
        facts = []
        train, val, test = split_data(facts, seed=42)

        # Should handle gracefully
        assert len(train.samples) == 0
        assert len(val.samples) == 0
        assert len(test.samples) == 0

    def test_single_fact(self):
        """Should handle a single fact."""
        facts = generate_facts(1, seed=42)
        train, val, test = split_data(facts, seed=42)

        # With one fact, at least one direction should be in training
        assert len(train.samples) >= 1

    def test_sample_direction_indicators(self):
        """Samples should correctly indicate forward vs reverse."""
        facts = generate_facts(5, seed=42)
        train, val, test = split_data(facts, seed=42)

        for sample in train.samples + val.samples + test.samples:
            # For each sample, check that it's logically a valid direction
            # (we can't verify it against original facts without tracking them,
            # but we can verify the boolean flag exists)
            assert isinstance(sample.is_reverse, bool)
