# Fact Retrieval Reversibility Experiment

This experiment investigates whether simple neural networks can learn **bidirectional** fact lookups on randomly generated data.

## Research Question

Can neural networks learn to perform bidirectional lookups of facts they've memorized? That is, if trained on pairs like (A→B), can they reliably predict both A→B and B→A?

## Experiment Design

### Data Generation

1. Generate N facts by:
   - Creating a list of numbers 0 to 2N-1
   - Randomly shuffling the list
   - Splitting in half to create N pairs

2. For each pair (a, b), we treat it as a **bidirectional fact**: the model should predict b from a AND predict a from b

### Data Split

- **Training set**: Contains at least one direction of each fact
- **Validation set**: Samples not in training (may contain reverse directions)
- **Test set**: Evaluation on unseen samples

This split strategy ensures:
- The model has seen every fact in at least one direction during training
- We can still evaluate generalization to unseen orderings/contexts

### Models Tested

Three simple fully-connected models are trained:

1. **Single Layer (Full)**: 1 hidden layer with `2N` hidden units (enough parameters to memorize all data)
2. **Single Layer (Half)**: 1 hidden layer with `N` hidden units (half the capacity)
3. **Two Layers**: 2 hidden layers with `N` units each

Each model:
- Takes one-hot encoded input (vocabulary size = 2N)
- Outputs logits for 2N classes
- Uses cross-entropy loss
- Is trained to classify the paired number

### Hypothesis

**The model will struggle with bidirectional lookups** - even with enough capacity to memorize, networks may fail to learn symmetric associations from random data. The model might learn a→b well but fail on b→a, suggesting it doesn't form true bidirectional knowledge.

## Installation

Using `uv`:

```bash
uv sync
```

## Running the Experiment

```bash
python -m fact_reversal.main --num_facts 10 --epochs 100
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
fact_reversal/
├── __init__.py
├── data.py           # Fact generation and data splitting
├── models.py         # Neural network architectures
├── train.py          # Training and evaluation logic
└── main.py           # Entry point for the experiment

tests/
├── test_data.py      # Tests for data generation
├── test_models.py    # Tests for model architectures
└── test_integration.py  # End-to-end tests
```

## Expected Results

We expect to observe:
- Models with sufficient capacity may memorize individual facts but struggle with bidirectionality
- The full-capacity model should perform better than the half-capacity model on forward lookups
- The two-layer model's performance will be interesting - it may show whether depth helps
- Likely failure pattern: High accuracy on seen directions, low accuracy on reverse directions
