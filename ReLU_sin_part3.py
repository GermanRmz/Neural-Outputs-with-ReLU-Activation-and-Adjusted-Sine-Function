import numpy as np
import matplotlib.pyplot as plt

w, w_02, w_21, w_22, w_71, w_72 = 6, -1, 3.5, -1, 0, 0    # weights
b, b_02, b_21, b_22, b_71, b_72 = 0, .7, -.42, .27, 0, 1    # biases
inputs = np.linspace(0, 1, 30)  # input values (from -2 to 2)

# Compute outputs for the neurons fisrt row
x_values = w * inputs + b  # Weighted sum
outputs = np.maximum(0, x_values)  # ReLU activation
outputs_inp = w_02 * outputs + b_02 
outputs_02 = np.maximum(0, outputs_inp)

# Second neuron pair
x_values_2 = w_21 * inputs + b_21  # Weighted sum
outputs_2 = np.maximum(0, x_values_2)  # ReLU activation
outputs_inp_2 = w_22 * outputs_2 + b_22 
outputs_22 = np.maximum(0, outputs_inp_2)

# last neuron pair
x_values_7 = w_71 * inputs + b_71  # Weighted sum
outputs_7 = np.maximum(0, x_values_7)  # ReLU activation
outputs_inp_7 = w_72 * outputs_7 + b_72 
outputs_72 = np.maximum(0, outputs_inp_7)

#output
total_output= outputs_02 * -1 + outputs_22 * -1 + outputs_72 * .97

# Compute sine function
sine_outputs = np.sin(2 * np.pi * inputs)  # Adjusted to range [-1, 1]

# Plot
plt.figure(figsize=(8, 5))
# Neuron outputs
#plt.plot(inputs, outputs, marker='o', label='First Neuron Output', color='red')
plt.plot(inputs, total_output, marker='o', label='Second Neuron Output', color='blue')
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
