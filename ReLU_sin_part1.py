import numpy as np
import matplotlib.pyplot as plt

w, w_2 = 6, -1    # weights
b, b_2 = 0, .7    # biases
inputs = np.linspace(0, 2, 30)  # input values (from -2 to 2)

# Compute outputs for the neurons
x_values = w * inputs + b  # Weighted sum
outputs = np.maximum(0, x_values)  # ReLU activation
outputs_inp = w_2 * outputs + b_2 
outputs_2 = np.maximum(0, outputs_inp)

# Compute sine function
sine_outputs = np.sin(inputs * np.pi)  # Adjusted to range [-1, 1]

# Plot
plt.figure(figsize=(8, 5))
# Neuron outputs
#plt.plot(inputs, outputs, marker='o', label='First Neuron Output', color='red')
plt.plot(inputs, -outputs_2, marker='o', label='Second Neuron Output', color='blue')
# Sine function
plt.plot(inputs, sine_outputs, label='Sine Function', color='green', linestyle='--')

# Reference lines and labels
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line
plt.title('Neural Outputs with ReLU Activation and Adjusted Sine Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(alpha=0.3)
plt.legend()
plt.show()
