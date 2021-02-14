import numpy as np


# Softmax activation
class ActivationSoftmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities**
        # included a subtraction of the largest of the inputs before we did the exponentiation
        # very large numbers (“exploding” values) tend to cause issues similar to dead (0) neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities