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
- **Test set**: Evaluation on unseen samples (typically the reverse direction of trained facts)

This split strategy ensures:
- The model has seen every fact in at least one direction during training
- We can evaluate generalization to the reverse direction

### Models Tested

#### Standard MLP Models

These models use separate weight matrices for input and output transformations:

1. **SingleLayerFull**: 1 hidden layer with `2N` hidden units (full capacity)
2. **SingleLayerHalf**: 1 hidden layer with `N` hidden units (half capacity)
3. **SingleLayerQuarter/Eighth/Sixteenth/Tiny**: Progressively smaller capacities
4. **TwoLayerNet**: 2 hidden layers with `N` units each

#### Symmetric Architecture Models

These models use shared embeddings between input and output, enabling bidirectional generalization:

5. **TiedWeights**: Autoencoder-style with `W_output = W_input.T` (tied weights)
6. **EmbeddingDotProduct**: Output logits = dot product of input embedding with all embeddings
7. **EmbeddingDotProductMasked**: Same as above, but masks self-predictions (diagonal)
8. **EmbeddingCosine**: Uses cosine similarity (normalized embeddings) with learnable temperature
9. **EmbeddingCosineMasked**: Cosine similarity with self-prediction masking

All models:
- Take one-hot encoded input (vocabulary size = 2N)
- Output logits for 2N classes
- Use cross-entropy loss
- Are trained with Adam optimizer

## Key Findings

### Standard MLPs Fail to Generalize

Standard MLP architectures achieve high training accuracy but **0% test accuracy** on reverse directions. They memorize unidirectional mappings without learning symmetric relationships.

### Self-Similarity is the Problem

The basic embedding models (`EmbeddingDotProduct`, `EmbeddingCosine`) also fail because `embed(a) · embed(a)` (self-similarity) is always maximal, causing the model to predict the input itself.

### Masked Embeddings Solve Bidirectionality

The masked variants (`EmbeddingDotProductMasked`, `EmbeddingCosineMasked`) achieve **100% test accuracy** by:
1. Using shared embeddings (symmetric by design)
2. Masking self-predictions to prevent trivial solutions

This demonstrates that bidirectional fact learning is possible with the right architectural inductive bias.

## Installation

Using `uv`:

```bash
uv sync
```

## Running the Experiment

Run all models:
```bash
python -m fact_reversal.main --num_facts 10 --epochs 100
```

Run specific models:
```bash
python -m fact_reversal.main --models EmbeddingDotProductMasked EmbeddingCosineMasked --epochs 200
```

Available models: `SingleLayerFull`, `SingleLayerHalf`, `SingleLayerQuarter`, `SingleLayerEighth`, `SingleLayerSixteenth`, `SingleLayerTiny`, `TwoLayerNet`, `TiedWeights`, `EmbeddingDotProduct`, `EmbeddingDotProductMasked`, `EmbeddingCosine`, `EmbeddingCosineMasked`

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

## Example Results

```
Model                     Train Acc    Test Acc
--------------------------------------------
SingleLayerFull           1.0000       0.0000
SingleLayerHalf           1.0000       0.0000
EmbeddingDotProduct       0.7143       0.0000
EmbeddingDotProductMasked 1.0000       1.0000  ✓
EmbeddingCosineMasked     1.0000       1.0000  ✓
```
