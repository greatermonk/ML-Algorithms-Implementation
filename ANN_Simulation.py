"""
*Practical no 09:- Implement & Simulate Feed Forward Neural Networks, Backpropagation.
"""
import numpy as np

# =============================================================================
# Part 1: The Simplest Neural Network - A Single Perceptron
# =============================================================================

class Perceptron:
    """
    A simple Perceptron class to model a single neuron.

    This Perceptron can learn linearly separable problems like AND, OR gates.
    """
    def __init__(self, num_inputs, learning_rate=0.01):
        """
        Initializes the Perceptron.

        Args:
            num_inputs (int): The number of input features (e.g., 2 for an AND gate).
            learning_rate (float): The step size for weight updates during training.
        """
        # Initialize weights and bias. Weights are initialized to zero.
        # In more complex networks, random initialization is crucial.
        self.weights = np.zeros(num_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def _step_function(self, z):
        """The activation function. Returns 1 if input is >= 0, else 0."""
        return 1 if z >= 0 else 0

    def predict(self, inputs):
        """
        Performs the feed-forward calculation to make a prediction.

        Args:
            inputs (np.array): A single input sample.

        Returns:
            int: The binary prediction (0 or 1).
        """
        # Step 1: Calculate the weighted sum (z)
        # z = w · x + b
        z = np.dot(inputs, self.weights) + self.bias

        # Step 2: Apply the activation function
        return self._step_function(z)


    def train(self, training_inputs, labels, epochs=100):
        """
        Trains the perceptron by adjusting weights and bias based on errors.

        Args:
            training_inputs (np.array): A 2D array of training samples.
            labels (np.array): The corresponding true labels for the training samples.
            epochs (int): The number of times to loop through the entire dataset.
        """
        print("--- Starting Perceptron Training ---")
        for epoch in range(epochs):
            num_errors = 0
            for inputs, label in zip(training_inputs, labels):
                # Make a prediction
                predictions = self.predict(inputs)

                # Calculate the error (true label - prediction)
                error = label - predictions

                if error != 0:
                    num_errors += 1
                    # Update weights and bias using the Perceptron learning rule:
                    # w_new = w_old + learning_rate * error * input
                    # b_new = b_old + learning_rate * error
                    update = self.learning_rate * error
                    self.weights += update * inputs
                    self.bias += update

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Errors: {num_errors}")
            if num_errors == 0:
                print(f"Training successful! Converged at epoch {epoch + 1}.")
                break
        print("--- Perceptron Training Finished ---")


# =============================================================================
# Part 2: The Multi-Layer Perceptron (MLP)
# =============================================================================

class MLP:
    """
    A Multi-Layer Perceptron with one hidden layer.

    This MLP is designed to solve non-linearly separable problems like the XOR gate.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the MLP's weights and biases.

        Args:
            input_size (int): Number of neurons in the input layer (e.g., 2 for XOR).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in the output layer (e.g., 1 for XOR).
        """
        # Initialize weights with small random values to break symmetry
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, z):
        """The Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        """Derivative of the Sigmoid function, where 'a' is the activated output."""
        return a * (1 - a)

    def forward(self, x):
        """
        Performs the feed-forward pass through the network.

        Args:
            x (np.array): The input data.

        Returns:
            np.array: The final output of the network.
        """
        # Step 1: From Input Layer to Hidden Layer
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self._sigmoid(self.z1)  # Activation of hidden layer

        # Step 2: From Hidden Layer to Output Layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        y_hat = self._sigmoid(self.z2)  # Final prediction

        return y_hat


    def backward(self, x, y, y_hat, learning_rate):
        """
        Performs the backpropagation step to update weights and biases.

        Args:
            x (np.array): Input data.
            y (np.array): True labels.
            y_hat (np.array): Predicted labels from the forward pass.
            learning_rate (float): The step size for updates.
        """
        # Calculate the error
        error = y - y_hat

        # --- Gradients for the Output Layer (W2, b2) ---
        # The delta for the output layer is the error * the derivative of the activation
        delta_output = error * self._sigmoid_derivative(y_hat)

        # Gradient for W2 is the dot product of the hidden layer's activation (transposed) and the output delta
        d_W2 = np.dot(self.a1.T, delta_output)
        # Gradient for b2 is the sum of the output deltas
        d_b2 = np.sum(delta_output, axis=0, keepdims=True)

        # --- Gradients for the Hidden Layer (W1, b1) ---
        # The delta for the hidden layer is the dot product of the output delta and W2 (transposed),
        # multiplied element-wise by the derivative of the hidden layer's activation
        delta_hidden = np.dot(delta_output, self.w2.T) * self._sigmoid_derivative(self.a1)

        # Gradient for W1
        d_W1 = np.dot(x.T, delta_hidden)
        # Gradient for b1
        d_b1 = np.sum(delta_hidden, axis=0, keepdims=True)

        # --- Update weights and biases ---
        self.w1 += learning_rate * d_W1
        self.b1 += learning_rate * d_b1
        self.w2 += learning_rate * d_W2
        self.b2 += learning_rate * d_b2


    def train(self, x, y, epochs=10000, learning_rate=0.1):
        """
        Trains the MLP using forward and backward passes.

        Args:
            x (np.array): Training data.
            y (np.array): True labels.
            epochs (int): Number of training iterations.
            learning_rate (float): The learning rate.
        """
        print("\n--- Starting MLP Training ---")
        for epoch in range(epochs):
            # Perform a full forward and backward pass
            y_hat = self.forward(x)
            self.backward(x, y, y_hat, learning_rate)

            # Print the loss every 1000 epochs to monitor training
            if (epoch + 1) % 1000 == 0:
                loss = np.mean((y - y_hat) ** 2)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
        print("--- MLP Training Finished ---")

# =============================================================================
# Main execution block
# =============================================================================

def main() -> None:
    # --- DEMONSTRATION 1: PERCEPTRON FOR AND GATE ---
    print("=" * 50)
    print("DEMONSTRATION 1: PERCEPTRON FOR AND GATE")
    print("=" * 50)

    # 1. Create synthetic data for the AND gate
    perceptron_inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    perceptron_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # 2. Create and train the Perceptron
    perceptron_network = Perceptron(num_inputs=3)
    perceptron_network.train(perceptron_inputs, perceptron_labels)

    # 3. Test the trained Perceptron
    print("\nTesting the trained Perceptron for AND logic:")
    for test_input in perceptron_inputs:
        prediction = perceptron_network.predict(test_input)
        print(f"Input: {test_input} -> Prediction: {prediction}")
    print(f"\nLearned Weights: {perceptron_network.weights}")
    print(f"Learned Bias: {perceptron_network.bias}")

    # --- DEMONSTRATION 2: MLP FOR XOR GATE ---
    print("\n\n" + "=" * 50)
    print("DEMONSTRATION 2: MLP FOR XOR GATE")
    print("=" * 50)

    # 1. Create simple binary data for the X-OR gate
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Labels need to be a column vector for matrix operations
    xor_labels = np.array([[0], [1], [1], [0]])

    # 2. Create and train the MLP
    # The number of hidden neurons (e.g., 4) is a hyperparameter you can tune.
    mlp = MLP(input_size=2, hidden_size=4, output_size=1)
    mlp.train(xor_inputs, xor_labels, epochs=10000, learning_rate=0.1)

    # 3. Test the trained MLP
    print("\nTesting the trained MLP for XOR logic:")
    predictions = mlp.forward(xor_inputs)
    # Apply a threshold of 0.5 to get binary output
    binary_predictions = (predictions > 0.5).astype(int)

    for i in range(len(xor_inputs)):
        print(
            f"Input: {xor_inputs[i]} -> Raw Output: {predictions[i][0]:.4f} -> Prediction: {binary_predictions[i][0]}")


if __name__=="__main__":
    main()