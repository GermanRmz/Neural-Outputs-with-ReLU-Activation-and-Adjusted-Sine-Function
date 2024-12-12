# Updated parameters
import numpy as np
import matplotlib.pyplot as plt

w, w_2 = -1, 1    # weight
b, b_2 = 0.5, 1  # bias
inputs = np.linspace(-1, 1, 30)  # input values

# Compute outputs
x_values = w * inputs + b  # Weighted sum
outputs = np.maximum(0, x_values)  # ReLU activation
outputs_inp = w_2 * outputs + b_2 
outputs_2 = np.maximum(0, outputs_inp)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(inputs, outputs_2, marker='o', label='ReLU output', color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line
plt.title('Neural Output with ReLU Activation (Updated Parameters)')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(alpha=0.3)
plt.legend()
plt.show()
