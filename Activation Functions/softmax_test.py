from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
from Classes.Layer_Dense import LayerDense
from Classes.Activation_Relu import ActivationReLU
from Classes.Softmax_Activation import ActivationSoftmax

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = LayerDense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = ActivationSoftmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])
