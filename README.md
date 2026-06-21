# Models From Scratch
Implementing machine learning and deep learning models from scratch using only NumPy/PyTorch tensors — no PyTorch autograd, no TensorFlow, no sklearn for the core logic. Every forward pass, every gradient, every weight update written by hand.

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
    ├── auto_grad_modified.py
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

### Custom Autograd Engine (`auto_grad_modified.py`)
A tensor-level automatic differentiation engine built from scratch — no PyTorch autograd.

- Every op registers its own backward pass into a computation graph (`back_list`)
- Supported ops: `mat_mul`, `mat_add`, `add`, `multiply`, `divide`, `subtract`, `scale`, `sqrt`, `relu`, `tanh`, `sigmoid`, `softmax`, `concat`, `embed`, `layer_norm`, `softmax_cross_entropy`
- Gradients derived by hand for every operation including layer norm (fused forward+backward with closure) and embedding lookup (indexed gradient accumulation)
- `backward()` replays the graph in reverse; `step()` applies SGD; `zero_grad()` and `clear()` reset state

### Transformer (`transformer.py`)
Character-level Transformer language model wired to the custom autograd engine.

- Pre-LN architecture (layer norm before attention and FFN)
- 2-head causal self-attention
- Sinusoidal positional encoding
- Feed-forward block with ReLU
- Causal masking (upper triangular -inf mask)
- Character-level tokenization and generation
- Loss: ~3.3 (random) → **1.0** after training

Every gradient — through attention, layer norm, feed-forward, projection, and embeddings — flows through the custom autograd engine. No PyTorch autograd anywhere.

### RNN (`rnn.py`)
Recurrent neural network built from scratch.

- Forward pass through time
- Backpropagation through time (BPTT)
- Manual gradient computation at every timestep

---

## Why from scratch?
Using a library like PyTorch is fast. Understanding *why* it works requires doing it without one. This repo is the foundation for deeper work in ML systems, custom CUDA kernels, and inference engine development.

---

## Requirements
```
torch
numpy
```

---

## Usage

For the Transformer, provide a `text.txt` file in `DL/` and run:
```bash
python DL/transformer.py
```
Training prints loss every 100 epochs. After training, `predict()` generates new text character by character.

---

## Results
### Transformer — Training Loss
![Training results](https://github.com/user-attachments/assets/3b730865-179a-480b-808e-e80ceadb95a0)
