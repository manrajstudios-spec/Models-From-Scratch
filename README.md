# Models From Scratch

Implementing machine learning and deep learning models from scratch using only NumPy — no PyTorch, no TensorFlow, no sklearn for the core logic. Every forward pass, every gradient, every weight update written by hand.

---

## Structure

```
Models_From_Scratch/
├── ml/
│   ├── linear_regression_scratch.py
│   ├── logistic_regression_scratch.py
│   ├── KNN.py
│   ├── Decision_Tree_Classifier_Scratch.py
│   ├── neural_network.py
│   └── neural_network_scratch.py
└── DL/
    ├── rnn.py
    ├── transformer.py
    └── text.txt
```

---

## ML — Classical Models (`ml/`)

| Model | What's implemented |
|---|---|
| Linear Regression | Gradient descent, MSE loss |
| Logistic Regression | Sigmoid, binary cross-entropy, gradient descent |
| K-Nearest Neighbors | Euclidean distance, majority vote |
| Decision Tree | Gini impurity, recursive splitting |
| Neural Network | Forward pass, backpropagation, SGD |

---

## DL — Deep Learning (`DL/`)

### RNN (`rnn.py`)
Recurrent neural network built from scratch.
- Forward pass through time
- Backpropagation through time (BPTT)
- Manual gradient computation at every timestep

### Transformer (`transformer.py`)
Character-level Transformer language model — the most complex model in this repo.
- Multi-head causal self-attention (4 heads)
- Manual backprop through the softmax Jacobian
- Sinusoidal positional encoding (vectorized)
- Residual connections
- Feed-forward block with ReLU
- Causal masking (upper triangular -inf mask)
- Character-level tokenization and generation

Every gradient — through attention, through the feed-forward block, through the projection layer, through embeddings — is computed by hand. No autograd.

---

## Why from scratch?

Using a library like PyTorch is fast. Understanding *why* it works requires doing it without one. This repo is the foundation for deeper work in ML systems and architecture research.

---

## Requirements

```
numpy
```

---

## Usage

For the Transformer, provide a `text.txt` file in `DL/` and run:

```bash
python DL/transformer.py
```

Training prints loss every 100 epochs. After training, `predict()` generates new text character by character.

#Results Of Transfomer
<img width="1395" height="511" alt="Screenshot From 2026-06-15 20-24-40" src="https://github.com/user-attachments/assets/3b730865-179a-480b-808e-e80ceadb95a0" />

