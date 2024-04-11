import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for the hidden and output layers
        self.weights_hidden = np.zeros((input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_output = np.zeros((hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def forward(self, inputs):
        # Forward pass through the network
        hidden_layer_input = np.dot(inputs, self.weights_hidden) + self.bias_hidden
        hidden_layer_output = self.relu(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.bias_output

        return output_layer_input


# Example usage:
input_size = 2
hidden_size = 3
output_size = 2

# Create a neural network
model = NeuralNetwork(input_size, hidden_size, output_size)

# Take weights for hidden layer from the user
print("Enter weights for the hidden layer:")
for i in range(input_size):
    for j in range(hidden_size):
        model.weights_hidden[i, j] = float(input(f"Enter weight W_{i + 1}{j + 1}: "))

# Take biases for hidden layer from the user
print("Enter biases for the hidden layer:")
for j in range(hidden_size):
    model.bias_hidden[0, j] = float(input(f"Enter bias B_{j + 1}: "))

# Take weights for output layer from the user
print("Enter weights for the output layer:")
for i in range(hidden_size):
    for j in range(output_size):
        model.weights_output[i, j] = float(input(f"Enter weight W_{i + 1}{j + 1}: "))

# Take biases for output layer from the user
print("Enter biases for the output layer:")
for j in range(output_size):
    model.bias_output[0, j] = float(input(f"Enter bias B_{j + 1}: "))

# Example 1: Using pre-stored inputs
# pre_stored_inputs = np.array([[0.5, 0.3], [0.8, 0.2]])
# output = model.forward(pre_stored_inputs)
# print("Output using pre-stored inputs:")
# print(output)

# Example 2: Taking inputs from the user
user_inputs = np.array([[float(input("Enter input 1: ")), float(input("Enter input 2: "))]])
output = model.forward(user_inputs)
print("Output using user inputs:")
print(output)
