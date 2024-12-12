import numpy as np
import matplotlib.pyplot as plt

# Pesos y sesgos para cada par de neuronas
weights = [6, -1,   3.5, -1,        -3.5, -1,     5.3, -1,   5.3, -1,    0,  0,     0, 0,    0, 0 ]
biases = [0, 0.7,   -0.42, 0.27,    1.35, 0.3,    -2, .7,   -2.65, .7,    0,  0,     0, 0,    0, 1 ]
scaling_factors = [-1, -1, -1, 1, 1, 1, 1, -.45]  # Factores de escala para la salida de cada par

inputs = np.linspace(0, 1, 30)  # Valores de entrada

# Inicializar la salida total
total_output = np.zeros_like(inputs)

# Calcular las salidas para cada par de neuronas
for i in range(0, len(weights), 2):
    w1, w2 = weights[i], weights[i + 1]  # Pesos para el par actual
    b1, b2 = biases[i], biases[i + 1]  # Sesgos para el par actual
    scale = scaling_factors[i // 2]  # Factor de escala para este par

    # Neurona 1
    x_values_1 = w1 * inputs + b1
    outputs_1 = np.maximum(0, x_values_1)

    # Neurona 2
    x_values_2 = w2 * outputs_1 + b2
    outputs_2 = np.maximum(0, x_values_2)

    # Agregar la salida escalada al total
    total_output += outputs_2 * scale

# Calcular la función seno
sine_outputs = np.sin(2 * np.pi * inputs)  # Ajustada al rango [-1, 1]

# Graficar
plt.figure(figsize=(8, 5))

# Salida total de la red neuronal
plt.plot(inputs, total_output, marker='o', label='Neural Network Output', color='blue')

# Función seno
plt.plot(inputs, sine_outputs, label='Sine Function', color='green', linestyle='--')

# Líneas de referencia y etiquetas
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Línea de referencia
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Línea de referencia
plt.title('Neural Outputs with ReLU Activation and Adjusted Sine Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(alpha=0.3)
plt.legend()
plt.show()
