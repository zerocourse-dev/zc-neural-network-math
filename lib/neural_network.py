"""
Neural Network Math from Scratch

A ZeroCourse capstone project for Q13: Integrating Linear Algebra, Calculus, and Probability.

Build a mini neural network library using only NumPy. No PyTorch, no TensorFlow, no sklearn.

Implement the functions and classes below. Each has a description of what it should do.
Run `python -m pytest tests/` to check your solutions.

Suggested implementation order:
  1. Activation functions (relu, sigmoid, softmax)
  2. Loss functions (mse_loss, cross_entropy_loss)
  3. Helper functions (one_hot_encode, generate_spiral_data)
  4. Layer (forward, then backward)
  5. NeuralNetwork (forward, backward, predict)
  6. Optimizers (SGD, then MomentumSGD)
  7. Trainer (train_step, train, evaluate)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation Functions (module-level, operate on numpy arrays)
# ---------------------------------------------------------------------------

def relu(x):
    """Element-wise ReLU: max(0, x).

    Examples:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 2])

    Args:
        x (np.ndarray): Input array of any shape.

    Returns:
        np.ndarray: Array with same shape, negative values replaced by 0.
    """
    raise NotImplementedError("Implement relu")


def relu_derivative(x):
    """Derivative of ReLU: 1 where x > 0, else 0.

    Examples:
        >>> relu_derivative(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 1])

    Args:
        x (np.ndarray): Input array (pre-activation values).

    Returns:
        np.ndarray: Array of 0s and 1s with same shape as x.
    """
    raise NotImplementedError("Implement relu_derivative")


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x)).

    Examples:
        >>> sigmoid(np.array([0]))
        array([0.5])

    Args:
        x (np.ndarray): Input array of any shape.

    Returns:
        np.ndarray: Values in range (0, 1) with same shape.
    """
    raise NotImplementedError("Implement sigmoid")


def sigmoid_derivative(x):
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x)).

    Examples:
        >>> sigmoid_derivative(np.array([0]))
        array([0.25])

    Args:
        x (np.ndarray): Input array (pre-activation values).

    Returns:
        np.ndarray: Derivative values with same shape.
    """
    raise NotImplementedError("Implement sigmoid_derivative")


def softmax(x):
    """Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x))).

    Applied along the last axis. Output sums to 1 along last axis.

    Examples:
        >>> softmax(np.array([1.0, 2.0, 3.0]))  # sums to 1.0
        array([0.09003057, 0.24472847, 0.66524096])

    Args:
        x (np.ndarray): Input array. For batches, shape is (batch_size, num_classes).

    Returns:
        np.ndarray: Probability distribution(s) with same shape, summing to 1 along last axis.
    """
    raise NotImplementedError("Implement softmax")


# ---------------------------------------------------------------------------
# Loss Functions (module-level)
# ---------------------------------------------------------------------------

def mse_loss(predictions, targets):
    """Mean Squared Error loss: mean((predictions - targets)^2).

    Examples:
        >>> mse_loss(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        0.0
        >>> mse_loss(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        1.0

    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): True values.

    Returns:
        float: Scalar loss value.
    """
    raise NotImplementedError("Implement mse_loss")


def mse_loss_derivative(predictions, targets):
    """Derivative of MSE loss: 2 * (predictions - targets) / n.

    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): True values.

    Returns:
        np.ndarray: Gradient with same shape as predictions.
    """
    raise NotImplementedError("Implement mse_loss_derivative")


def cross_entropy_loss(predictions, targets):
    """Cross-entropy loss for one-hot encoded targets.

    Formula: -mean(sum(targets * log(predictions + epsilon)))

    Args:
        predictions (np.ndarray): Predicted probabilities, shape (n, classes).
        targets (np.ndarray): One-hot encoded targets, shape (n, classes).

    Returns:
        float: Scalar loss value.
    """
    raise NotImplementedError("Implement cross_entropy_loss")


def cross_entropy_loss_derivative(predictions, targets):
    """Derivative of cross-entropy loss (with softmax): (predictions - targets) / n.

    This is the simplified gradient when softmax is the final activation.

    Args:
        predictions (np.ndarray): Predicted probabilities, shape (n, classes).
        targets (np.ndarray): One-hot encoded targets, shape (n, classes).

    Returns:
        np.ndarray: Gradient with same shape as predictions.
    """
    raise NotImplementedError("Implement cross_entropy_loss_derivative")


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Layer:
    """A single fully-connected (dense) neural network layer.

    Uses He initialization for weights and zeros for biases.

    Attributes:
        weights (np.ndarray): Weight matrix, shape (input_size, output_size).
        biases (np.ndarray): Bias vector, shape (1, output_size).
        activation (str): Name of activation function ("relu", "sigmoid", or "softmax").
        grad_weights (np.ndarray): Gradient of loss w.r.t. weights (set during backward).
        grad_biases (np.ndarray): Gradient of loss w.r.t. biases (set during backward).
    """

    def __init__(self, input_size, output_size, activation="relu", seed=42):
        """Initialize layer with He initialization.

        He initialization: weights = randn * sqrt(2 / input_size)
        Biases initialized to zeros.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (neurons).
            activation (str): Activation function name ("relu", "sigmoid", "softmax").
            seed (int): Random seed for reproducibility.
        """
        raise NotImplementedError("Implement Layer.__init__")

    def forward(self, x):
        """Forward pass: z = x @ weights + bias, then apply activation.

        Stores x (input) and z (pre-activation) for use in backward pass.

        Args:
            x (np.ndarray): Input data, shape (batch_size, input_size).

        Returns:
            np.ndarray: Activated output, shape (batch_size, output_size).
        """
        raise NotImplementedError("Implement Layer.forward")

    def backward(self, grad_output):
        """Backward pass: compute and store gradients.

        Computes:
            - grad_weights: gradient of loss w.r.t. weights
            - grad_biases: gradient of loss w.r.t. biases
            - grad_input: gradient of loss w.r.t. input (to pass to previous layer)

        Args:
            grad_output (np.ndarray): Gradient flowing from the next layer,
                shape (batch_size, output_size).

        Returns:
            np.ndarray: Gradient w.r.t. input, shape (batch_size, input_size).
        """
        raise NotImplementedError("Implement Layer.backward")


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """A feedforward neural network composed of multiple layers.

    Examples:
        >>> net = NeuralNetwork([784, 128, 64, 10])  # 3-layer network
        >>> output = net.forward(np.random.randn(32, 784))  # batch of 32
        >>> output.shape
        (32, 10)

    Attributes:
        layers (list[Layer]): List of Layer objects.
    """

    def __init__(self, layer_sizes, activations=None, seed=42):
        """Create network from list of layer sizes.

        Default activations: "relu" for all hidden layers, "softmax" for the output layer.

        Args:
            layer_sizes (list[int]): Sizes of each layer, e.g. [784, 128, 64, 10].
            activations (list[str] | None): Activation for each layer (len = len(layer_sizes) - 1).
                If None, uses "relu" for hidden and "softmax" for output.
            seed (int): Random seed for reproducibility.
        """
        raise NotImplementedError("Implement NeuralNetwork.__init__")

    def forward(self, x):
        """Forward pass through all layers.

        Args:
            x (np.ndarray): Input data, shape (batch_size, input_features).

        Returns:
            np.ndarray: Network output, shape (batch_size, output_features).
        """
        raise NotImplementedError("Implement NeuralNetwork.forward")

    def backward(self, loss_gradient):
        """Backward pass through all layers in reverse order.

        Args:
            loss_gradient (np.ndarray): Gradient of loss w.r.t. network output.
        """
        raise NotImplementedError("Implement NeuralNetwork.backward")

    def predict(self, x):
        """Forward pass, return class prediction (argmax of output).

        Args:
            x (np.ndarray): Input data, shape (batch_size, input_features).

        Returns:
            np.ndarray: Predicted class indices, shape (batch_size,).
        """
        raise NotImplementedError("Implement NeuralNetwork.predict")


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class SGD:
    """Stochastic Gradient Descent optimizer.

    Updates weights and biases using: param -= learning_rate * gradient
    """

    def __init__(self, learning_rate=0.01):
        """Initialize SGD optimizer.

        Args:
            learning_rate (float): Step size for parameter updates.
        """
        raise NotImplementedError("Implement SGD.__init__")

    def update(self, layers):
        """Update weights and biases for all layers using stored gradients.

        Args:
            layers (list[Layer]): List of Layer objects with grad_weights and grad_biases set.
        """
        raise NotImplementedError("Implement SGD.update")


class MomentumSGD:
    """SGD with momentum optimizer.

    Maintains velocity for each parameter:
        velocity = momentum * velocity - learning_rate * gradient
        param += velocity
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initialize MomentumSGD optimizer.

        Velocities are initialized to zero on the first call to update().

        Args:
            learning_rate (float): Step size for parameter updates.
            momentum (float): Momentum factor (typically 0.9).
        """
        raise NotImplementedError("Implement MomentumSGD.__init__")

    def update(self, layers):
        """Update weights and biases using momentum.

        On first call, initializes velocity arrays to zeros matching each parameter's shape.

        Args:
            layers (list[Layer]): List of Layer objects with grad_weights and grad_biases set.
        """
        raise NotImplementedError("Implement MomentumSGD.update")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates training and evaluation of a neural network.

    Examples:
        >>> net = NeuralNetwork([2, 16, 3])
        >>> optimizer = SGD(learning_rate=0.1)
        >>> trainer = Trainer(net, optimizer, cross_entropy_loss, cross_entropy_loss_derivative)
        >>> result = trainer.train(x_train, y_train, epochs=50)
        >>> result["loss_history"][-1] < result["loss_history"][0]
        True
    """

    def __init__(self, network, optimizer, loss_func, loss_derivative):
        """Initialize trainer.

        Args:
            network (NeuralNetwork): The network to train.
            optimizer (SGD | MomentumSGD): Optimizer for parameter updates.
            loss_func (callable): Loss function (e.g., cross_entropy_loss).
            loss_derivative (callable): Derivative of loss function.
        """
        raise NotImplementedError("Implement Trainer.__init__")

    def train_step(self, x_batch, y_batch):
        """Perform one training step: forward, loss, backward, update.

        Args:
            x_batch (np.ndarray): Input batch, shape (batch_size, features).
            y_batch (np.ndarray): Target batch (one-hot), shape (batch_size, classes).

        Returns:
            float: Loss value for this batch.
        """
        raise NotImplementedError("Implement Trainer.train_step")

    def train(self, x_train, y_train, epochs=10, batch_size=32, seed=42):
        """Full training loop with mini-batches.

        Shuffles data each epoch using the provided seed (incremented per epoch).

        Args:
            x_train (np.ndarray): Training inputs, shape (n_samples, features).
            y_train (np.ndarray): Training targets (one-hot), shape (n_samples, classes).
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size.
            seed (int): Random seed for shuffling.

        Returns:
            dict: {"loss_history": list} where loss_history contains average loss per epoch.
        """
        raise NotImplementedError("Implement Trainer.train")

    def evaluate(self, x_test, y_test):
        """Evaluate network on test data.

        Args:
            x_test (np.ndarray): Test inputs.
            y_test (np.ndarray): Test targets (one-hot encoded).

        Returns:
            dict: {"accuracy": float, "loss": float}
        """
        raise NotImplementedError("Implement Trainer.evaluate")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def one_hot_encode(labels, num_classes):
    """Convert integer labels to one-hot encoded numpy array.

    Examples:
        >>> one_hot_encode(np.array([0, 1, 2]), 3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

    Args:
        labels (np.ndarray): Integer labels, shape (n,).
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded array, shape (n, num_classes).
    """
    raise NotImplementedError("Implement one_hot_encode")


def generate_spiral_data(n_points_per_class=100, n_classes=3, seed=42):
    """Generate a spiral dataset for classification testing.

    Creates n_classes interleaved spirals. Useful for testing nonlinear classifiers.

    Args:
        n_points_per_class (int): Number of data points per class.
        n_classes (int): Number of spiral arms / classes.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (x_data, y_labels) where:
            - x_data has shape (n_points_per_class * n_classes, 2)
            - y_labels has shape (n_points_per_class * n_classes,) with integer labels
    """
    raise NotImplementedError("Implement generate_spiral_data")
