# Neural Network Math from Scratch

A ZeroCourse capstone project for Q13: Integrating Linear Algebra, Calculus, and Probability.

## What You'll Build

Implement a mini neural network library using **only NumPy** — no PyTorch, no TensorFlow, no sklearn. You'll build every component from the ground up: activation functions, loss functions, layers with forward/backward passes, optimizers, and a training loop.

This project integrates everything from Quarter 1 Section 3:
- **Linear algebra** — matrix multiplications for forward/backward passes, weight initialization
- **Calculus** — gradients, chain rule, backpropagation
- **Probability** — softmax outputs as probability distributions, cross-entropy loss

### Functions and Classes

| Component | Description |
|-----------|-------------|
| `relu(x)` | Element-wise max(0, x) activation |
| `relu_derivative(x)` | 1 where x > 0, else 0 |
| `sigmoid(x)` | Logistic activation: 1 / (1 + exp(-x)) |
| `sigmoid_derivative(x)` | sigmoid(x) * (1 - sigmoid(x)) |
| `softmax(x)` | Numerically stable softmax (sums to 1) |
| `mse_loss(predictions, targets)` | Mean Squared Error loss |
| `mse_loss_derivative(predictions, targets)` | MSE gradient |
| `cross_entropy_loss(predictions, targets)` | Cross-entropy with one-hot targets |
| `cross_entropy_loss_derivative(predictions, targets)` | Simplified CE gradient (with softmax) |
| `Layer(input_size, output_size, activation, seed)` | Dense layer with He init, forward, backward |
| `NeuralNetwork(layer_sizes, activations, seed)` | Multi-layer feedforward network |
| `SGD(learning_rate)` | Stochastic Gradient Descent optimizer |
| `MomentumSGD(learning_rate, momentum)` | SGD with momentum |
| `Trainer(network, optimizer, loss_func, loss_derivative)` | Training loop orchestrator |
| `one_hot_encode(labels, num_classes)` | Integer labels to one-hot arrays |
| `generate_spiral_data(n_points_per_class, n_classes, seed)` | Spiral dataset generator |

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests (they will all fail initially):
   ```bash
   python -m pytest tests/ --tb=short
   ```

3. Open `lib/neural_network.py` and implement each function/class.

4. Run the tests again to check your progress:
   ```bash
   python -m pytest tests/ -v
   ```

## Suggested Implementation Order

1. **Activation functions** (`relu`, `sigmoid`, `softmax` and their derivatives) — pure math, no dependencies
2. **Loss functions** (`mse_loss`, `cross_entropy_loss` and derivatives) — also pure math
3. **Helper functions** (`one_hot_encode`, `generate_spiral_data`) — needed for testing later components
4. **Layer** — `__init__` first (He initialization), then `forward`, then `backward`
5. **NeuralNetwork** — `forward` and `predict` first, then `backward`
6. **Optimizers** — `SGD` first (simple), then `MomentumSGD`
7. **Trainer** — `train_step`, then `train`, then `evaluate`

## Tips

- **Start simple.** Get activation functions passing first — they're one-liners with NumPy.
- **He initialization:** `weights = rng.standard_normal((input_size, output_size)) * np.sqrt(2.0 / input_size)` where `rng = np.random.default_rng(seed)`.
- **Softmax stability:** Always subtract `max(x)` before exponentiating to avoid overflow.
- **Cross-entropy epsilon:** Add a tiny value (e.g., `1e-15`) inside `log()` to avoid `log(0)`.
- **Backpropagation:** The chain rule is your friend. For each layer: `grad_input = grad_output * activation_derivative @ weights.T` (simplified). Draw the computation graph if you get stuck.
- **Batch dimensions:** Keep track of shapes. Inputs are `(batch_size, features)`, weights are `(input_size, output_size)`.
- **Test incrementally:** Run specific test classes as you go: `python -m pytest tests/ -k "TestRelu" -v`

## Key Concepts

### Forward Pass
```
input -> [Layer 1: z = x @ W + b, a = relu(z)] -> [Layer 2: z = a @ W + b, a = softmax(z)] -> output
```

### Backward Pass (Backpropagation)
```
loss_gradient <- [Layer 2: compute dW, db, dx] <- [Layer 1: compute dW, db, dx] <- done
```

### He Initialization
Weights are drawn from N(0, sqrt(2/n_in)) to maintain variance through deep networks with ReLU activation.

### Mini-Batch Training
Each epoch shuffles data and splits into batches. For each batch: forward pass, compute loss, backward pass, update weights.

## Running Tests

```bash
python -m pytest tests/                          # Run all tests
python -m pytest tests/ -v                       # Verbose output
python -m pytest tests/ -k "TestLayer"           # Run only Layer tests
python -m pytest tests/ -k "TestIntegration"     # Run only integration tests
python -m pytest tests/ --tb=long                # Full tracebacks on failure
```
