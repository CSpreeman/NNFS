import numpy as np
# Values from the earlier previous when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]
# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponential values:')
print(exp_values)
# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponential values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))


# what does softmax achieve?
# Values from the previous output when we described
layer_outputs = [4.8, 1.21, 2.385]

# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercase
E = 2.71828182846  # you can also use math.e

# For each value in a vector, calculate the exponential value
exp_values = []

for output in layer_outputs:
    exp_values.append(E ** output)  # ** - power operator in Python

print('Exponential values:')
print(exp_values)

# Now normalize values
norm_base = sum(exp_values)  # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponential values:')
print(norm_values)
print('Sum of normalized values:', sum(norm_values))
