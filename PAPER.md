# Bidirectional Fact Retrieval in Neural Networks: The Role of Architectural Symmetry

## Abstract

We investigate whether neural networks can learn bidirectional associations from unidirectional training data. Given fact pairs (a, b), models trained only on the forward direction (a→b) are tested on the reverse direction (b→a). Standard multilayer perceptrons achieve near-perfect training accuracy but completely fail on reverse lookups (0% test accuracy), demonstrating that memorization does not imply bidirectional understanding. We show that this failure stems from asymmetric information flow in standard architectures, where input and output representations are decoupled. We introduce embedding-based architectures with shared representations and demonstrate that self-similarity masking is critical: masked embedding models achieve 100% accuracy on both directions, while unmasked variants fail due to trivial self-prediction. These results highlight how architectural inductive biases determine whether networks learn symmetric relationships or merely memorize directional mappings.

## 1. Introduction

Neural networks excel at memorizing training data, but memorization does not guarantee understanding. A network that perfectly predicts "Paris is the capital of France" may fail entirely when asked "What city is the capital of France?" This asymmetry raises a fundamental question: **when a network learns a fact in one direction, does it implicitly learn the reverse?**

This question has implications for knowledge representation in neural networks. If models store facts as unidirectional input-output mappings, they may require explicit training on every query direction. Alternatively, if models learn symmetric representations, knowledge of (a, b) might automatically transfer to (b, a).

We design a controlled experiment using randomly generated fact pairs with no semantic structure, isolating the architectural factors that enable or prevent bidirectional generalization.

## 2. Experimental Setup

### 2.1 Data Generation

We generate N random bidirectional facts as follows:

1. Create integers 0 to 2N-1
2. Randomly shuffle the list
3. Split into two halves of size N
4. Pair elements by index: fact_i = (first_half[i], second_half[i])

This produces N pairs where each integer appears in exactly one pair. The random pairing ensures no exploitable structure—the only way to predict b from a is to have learned the specific (a, b) association.

### 2.2 Train/Test Split

For each fact pair (a, b), we create two samples:
- Forward: input=a, target=b
- Reverse: input=b, target=a

The critical aspect of our split: **for approximately half the facts, only one direction appears in training, with the opposite direction held out for testing.** This tests whether learning a→b enables predicting b→a.

Specifically, for each fact:
- With 50% probability: only forward direction in training, reverse in test
- With 50% probability: only reverse direction in training, forward in test
- Remaining facts: both directions in training

### 2.3 Model Input/Output

All models receive one-hot encoded inputs of dimension 2N (vocabulary size) and output logits over 2N classes. Training uses cross-entropy loss with Adam optimizer.

## 3. Architectures Investigated

### 3.1 Standard MLPs

Standard multilayer perceptrons with separate weight matrices for input projection and output projection:

```
Input (one-hot) → W₁ → ReLU → W₂ → Output (logits)
```

We test variants with different hidden layer sizes (full capacity to 1/32 capacity) and depths (1-2 hidden layers).

**Key property:** W₁ and W₂ are independent parameters. The representation of "3 as input" (column 3 of W₁) has no structural relationship to "3 as output" (row 3 of W₂).

### 3.2 Tied Weights (Autoencoder-style)

An autoencoder-inspired architecture where the output weight matrix is the transpose of the input weight matrix:

```
Input → W → ReLU → Wᵀ → Output
```

**Key property:** The same weight vector represents an entity whether it appears as input or output, creating a shared embedding space.

### 3.3 Embedding with Dot Product

Each entity has a single learned embedding vector. Output logits are computed as dot products between the input embedding and all embeddings:

```
logits[j] = embed(input) · embed(j)
```

**Key property:** The similarity function is symmetric: embed(a)·embed(b) = embed(b)·embed(a). Learning that a and b should have high similarity when a is input automatically means they have high similarity when b is input.

### 3.4 Embedding with Cosine Similarity

Same as dot product, but embeddings are L2-normalized before computing similarity, with a learnable temperature parameter:

```
logits[j] = temperature × (embed(input)/‖embed(input)‖) · (embed(j)/‖embed(j)‖)
```

**Key property:** Similarity is based on angle rather than magnitude, preventing any single embedding from dominating due to large norm.

### 3.5 Masked Variants

For both dot product and cosine models, we add a "masked" variant that sets the self-prediction logit to negative infinity:

```
logits[input_index] = -∞
```

This prevents the model from predicting the input itself.

## 4. Results

### 4.1 Standard MLPs Fail Completely

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| SingleLayerFull | 100% | 0% |
| SingleLayerHalf | 100% | 0% |
| TwoLayerNet | 100% | 0% |

All MLP variants achieve perfect training accuracy but **zero test accuracy** on reverse directions. This is not a capacity issue—even heavily overparameterized models fail identically.

### 4.2 Basic Embedding Models Also Fail

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| TiedWeights | ~71% | 0% |
| EmbeddingDotProduct | ~71% | 0% |
| EmbeddingCosine | ~0% | 0% |

Surprisingly, the symmetric embedding architectures also fail. They even struggle to fit the training data, plateauing around 71% accuracy. The cosine variant barely learns at all.

### 4.3 Masked Embedding Models Succeed

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| EmbeddingDotProductMasked | 100% | **100%** |
| EmbeddingCosineMasked | 100% | **100%** |

The masked variants achieve perfect accuracy on both training and test sets, including the held-out reverse directions.

## 5. Analysis

### 5.1 Why MLPs Fail: Decoupled Representations

In a standard MLP, the input transformation W₁ and output transformation W₂ are independent:

- W₁ learns: "when input one-hot[3] is active, produce hidden pattern X"
- W₂ learns: "when hidden pattern X is active, output high logit for class 7"

This is a **unidirectional lookup table**. There is no mechanism connecting the representation of "3 as input" to "3 as output." The model learns a function f where f(3)=7, but this provides no information about f(7).

Even with unlimited capacity, the model has no reason to learn that (3,7) and (7,3) are related—they involve completely different weight paths.

### 5.2 Why Basic Embedding Models Fail: Self-Similarity Dominance

The embedding models have symmetric similarity by construction: embed(a)·embed(b) = embed(b)·embed(a). So why do they fail?

The problem is **self-similarity**. For any embedding:

```
embed(a) · embed(a) = ‖embed(a)‖² ≥ embed(a) · embed(b)
```

The dot product of a vector with itself is always maximal (for that vector). When we query with input a, the highest-scoring output is always a itself, regardless of training.

During training on (a→b):
- Gradient pushes embed(a) toward embed(b)
- But embed(a)·embed(a) remains the largest score
- The model predicts a→a, gets it wrong, but the correct answer b is the second-highest score

The model learns to make embed(a) and embed(b) similar, but can never overcome self-similarity. At test time, querying with b also predicts b→b instead of b→a.

For cosine similarity, self-similarity is exactly 1.0 (maximum possible), making it even harder to overcome.

### 5.3 Why Masked Embeddings Succeed

Masking removes self-predictions from consideration:

```
logits[input_index] = -∞
```

Now when querying with input a:
- Self-prediction a→a is impossible (masked out)
- The highest remaining score is embed(a)·embed(b)
- Model correctly predicts a→b

Crucially, the symmetric property now works as intended:
- Training on (a→b) pushes embed(a) and embed(b) to be similar
- At test time, querying b finds that embed(b)·embed(a) is high
- Model correctly predicts b→a

The masking removes the degenerate solution, allowing the symmetric structure to enable bidirectional generalization.

### 5.4 Why Tied Weights Also Fail

The tied-weight model (W_out = W_in.T) should theoretically enable sharing, but it includes a ReLU nonlinearity:

```
hidden = ReLU(W · input)
output = Wᵀ · hidden
```

The ReLU breaks symmetry. The hidden representation depends on which input units are active, and the sparsity pattern after ReLU differs between inputs. While weights are shared, the effective computation path is still asymmetric.

Additionally, without masking, the tied-weight model can still achieve high self-reconstruction scores, falling into a similar trap as the embedding models.

## 6. Discussion

### 6.1 Architectural Inductive Bias Matters

The standard MLP has no inductive bias toward symmetric relationships. Even with infinite capacity and training time, it cannot discover that learning a→b should inform b→a—there is no gradient signal connecting these.

The embedding architecture encodes symmetry structurally: the same parameters define both directions. But this alone is insufficient; the architecture must also prevent degenerate solutions.

### 6.2 The Self-Prediction Trap

Self-similarity is a subtle failure mode. The model is "trying" to learn the right associations (embeddings of paired entities become similar), but an architectural limitation (self-prediction is always optimal) prevents this from translating to correct predictions.

This suggests that when designing architectures for relational learning, one must consider not just whether the desired solution is representable, but whether degenerate solutions might be easier to reach.

### 6.3 Implications for Knowledge Representation

These results suggest that standard neural network architectures store facts as directional mappings rather than symmetric associations. A network trained on "Paris → France" does not automatically know "France → Paris."

For applications requiring bidirectional knowledge access (question answering, knowledge graphs, etc.), architectural choices matter:
1. **Shared embeddings** between query and answer spaces
2. **Preventing trivial solutions** like self-prediction
3. **Explicit symmetry** in the similarity function

### 6.4 Relationship to Knowledge Graph Embeddings

Our findings parallel work on knowledge graph embeddings (TransE, RotatE, etc.), which learn entity embeddings such that relationship patterns emerge from geometric operations. These methods typically:
- Use shared entity embeddings (symmetric by design)
- Model relations as transformations (translation, rotation)
- Employ negative sampling to prevent degenerate solutions

Our masked embedding model is a simplified version of this approach, demonstrating that the core principles apply even to arbitrary random associations.

## 7. Conclusion

We investigated whether neural networks can learn bidirectional fact associations from unidirectional training data. Our key findings:

1. **Standard MLPs cannot generalize bidirectionally**, regardless of capacity. They achieve 100% training accuracy but 0% test accuracy on reverse directions.

2. **Symmetric architectures alone are insufficient.** Embedding models with shared representations fail due to self-similarity dominance.

3. **Masking self-predictions enables perfect bidirectional generalization.** Masked embedding models achieve 100% accuracy on both trained and reverse directions.

The critical insight is that **architectural inductive bias determines whether networks learn symmetric relationships or directional mappings.** Simply having enough parameters to represent bidirectional knowledge does not mean the network will learn it—the architecture must structurally favor symmetric solutions and prevent degenerate alternatives.

## References

- Bordes, A., et al. (2013). Translating embeddings for modeling multi-relational data. *NeurIPS*.
- Sun, Z., et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR*.
- Petroni, F., et al. (2019). Language models as knowledge bases? *EMNLP*.
