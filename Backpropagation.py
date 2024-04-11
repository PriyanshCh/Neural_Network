import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_function):
        # Initialize weights and biases for the layers
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.weights = [np.zeros((layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

        # Take weights and biases from the user
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] = float(input(f"Enter weight W_{i + 1}{j + 1}{k + 1}: "))

        for i in range(len(self.biases)):
            for j in range(self.biases[i].shape[1]):
                self.biases[i][0, j] = float(input(f"Enter bias B_{i + 1}{j + 1}: "))

    def forward(self, inputs):
        # Forward pass through the network
        layer_output = inputs
        self.layer_inputs = []
        self.layer_outputs = [inputs]

        for i in range(len(self.layer_sizes) - 1):
            layer_input = np.dot(layer_output, self.weights[i]) + self.biases[i]
            layer_output = self.apply_activation(layer_input)

            self.layer_inputs.append(layer_input)
            self.layer_outputs.append(layer_output)

        return layer_output

    def apply_activation(self, x):
        # Apply the chosen activation function
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def apply_activation_derivative(self, x):
        # Calculate the derivative of the activation function
        if self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_function == 'sigmoid':
            sigmoid_x = self.apply_activation(x)
            return sigmoid_x * (1 - sigmoid_x)
        else:
            raise ValueError("Unsupported activation function")

    def mean_squared_error_loss(self, predictions, targets):
        # Mean Squared Error loss
        return np.mean((predictions - targets) ** 2)

    def cross_entropy_loss(self, predictions, targets):
        # Cross-entropy loss for binary classification
        epsilon = 1e-15  # Small value to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    def calculate_loss(self, predictions, targets):
        # Choose between mean squared error loss or cross-entropy loss
        if self.activation_function == 'sigmoid':
            return self.cross_entropy_loss(predictions, targets)
        else:
            return self.mean_squared_error_loss(predictions, targets)

    def backpropagation(self, inputs, targets, learning_rate):
        # Backpropagation algorithm to update weights and biases
        output_error = targets - self.layer_outputs[-1]
        for i in range(len(self.layer_sizes) - 2, -1, -1):
            error_delta = output_error * self.apply_activation_derivative(self.layer_inputs[i])
            weight_gradients = np.dot(self.layer_outputs[i].T, error_delta)
            self.weights[i] += learning_rate * weight_gradients
            self.biases[i] += learning_rate * np.sum(error_delta, axis=0, keepdims=True)
            output_error = np.dot(error_delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate, loss_function):
        # Train the model using gradient descent
        for epoch in range(epochs):
            predictions = self.forward(inputs)
            loss = self.calculate_loss(predictions, targets)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
            self.backpropagation(inputs, targets, learning_rate)

# Example usage:
num_layers = int(input("Enter the number of layers: "))
layer_sizes = [int(input(f"Enter the number of neurons in layer {i + 1}: ")) for i in range(num_layers)]

activation_function = input("Enter the activation function (relu or sigmoid): ")
model = NeuralNetwork(layer_sizes, activation_function)

# Example data (XOR problem)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Train the model
epochs = int(input("Enter the number of epochs: "))
learning_rate = float(input("Enter learning rate: "))
loss_function = input("Choose loss function (mse or cross_entropy): ")

model.train(inputs, targets, epochs, learning_rate, loss_function)

# Test the trained model
user_inputs = np.array([[float(input(f"Enter input {i + 1}: ")) for i in range(layer_sizes[0])]])
output = model.forward(user_inputs)
print("Output using user inputs:")
print(output)
