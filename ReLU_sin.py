import numpy as np
import matplotlib.pyplot as plt

# Pesos y sesgos para cada par de neuronas
weights = [6, -1,   3.5, -1,        -3.5, -1,     -6, -1,        -3.5, -1,    3.6,  -1,     6, -1,        0, 0 ]
biases = [0, 0.7,   -0.42, 0.27,    1.35, 0.3,    3.69, 1.37,   2.44, .27,    -2.9,  .27,     -5.3, 0.7,    0, 1 ]
scaling_factors = [-1, -1, -1, -1, -1, -1, -1, 1.94]  # Factores de escala para la salida de cada par

inputs = np.linspace(0, 1, 30)  # Valores de entrada

# Inicializar la salida total
total_output = np.zeros_like(inputs)

# Calcular las salidas para cada par de neuronas
neurons_outputs = []  # Para almacenar las salidas de las neuronas
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

    # Almacenar las salidas de las neuronas
    neurons_outputs.append((outputs_1, outputs_2))

    # Agregar la salida escalada al total
    total_output += outputs_2 * scale

# Calcular la función seno
sine_outputs = np.sin(2 * np.pi * inputs)  # Ajustada al rango [-1, 1]

# Agregar un nuevo input
new_input = 0.51
new_output_total = 0  # Salida total para el nuevo input
active_neurons = []   # Lista de neuronas activadas
active_outputs = []   # Salidas activadas
for i in range(0, len(weights), 2):
    w1, w2 = weights[i], weights[i + 1]
    b1, b2 = biases[i], biases[i + 1]
    scale = scaling_factors[i // 2]

    # Neurona 1
    x_value_1 = w1 * new_input + b1
    output_1 = np.maximum(0, x_value_1)

    # Neurona 2
    x_value_2 = w2 * output_1 + b2
    output_2 = np.maximum(0, x_value_2)

    # Checar activación
    if output_1 > 0:
        active_neurons.append(f'Layer 1, Neuron {i // 2 * 2 + 1}')
        active_outputs.append(1)
    else:
        active_outputs.append(0)

    if output_2 > 0:
        active_neurons.append(f'Layer 2, Neuron {i // 2 * 2 + 2}')
        active_outputs.append(1)
    else:
        active_outputs.append(0)

    # Agregar salida escalada
    new_output_total += output_2 * scale

# Graficar la salida total de la red neuronal y la función seno
plt.figure(figsize=(8, 5))

# Salida total de la red neuronal
plt.plot(inputs, total_output, marker='o', label='Neural Network Output', color='blue', zorder=1)

# Función seno
plt.plot(inputs, sine_outputs, label='Sine Function', color='green', linestyle='--', zorder=2)

# Punto de entrada nuevo (con un zorder alto para que esté encima)
plt.scatter(new_input, new_output_total, color='red', label='New Input', zorder=3)

# Líneas de referencia y etiquetas
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)  # Línea de referencia
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)  # Línea de referencia
plt.title('Neural Outputs with ReLU Activation and Adjusted Sine Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Mostrar neuronas activadas
print(f"Input: {new_input}")
print(f"Output: {new_output_total}")
print("Active neurons:")
for neuron in active_neurons:
    print(neuron)

# Graficar neuronas activadas
plt.figure(figsize=(8, 5))

# Mostrar puntos para cada neurona en ambas capas
for i in range(8):
    # Capa 1, neurona i+1
    if active_outputs[i] == 1:
        plt.scatter(1, i + 1, color='green', s=100, label=f'Neuron {i + 1} Activated' if i == 0 else "")
    else:
        plt.scatter(1, i + 1, color='blue', s=100, label=f'Neuron {i + 1} Inactivated' if i == 0 else "")

    # Capa 2, neurona i+9
    if active_outputs[i + 8] == 1:
        plt.scatter(2, i + 1, color='green', s=100, label=f'Neuron {i + 9} Activated' if i == 0 else "")
    else:
        plt.scatter(2, i + 1, color='blue', s=100, label=f'Neuron {i + 9} Inactivated' if i == 0 else "")

# Etiquetas y formato
plt.xlim(0.5, 2.5)
plt.ylim(0.5, 8.5)
plt.xticks([1, 2], ['Layer 1', 'Layer 2'])
plt.yticks(np.arange(1, 9), [f'Neuron {i}' for i in range(1, 9)])
plt.title(f'Neuron Activation for Input {new_input}')
plt.xlabel('Layers')
plt.ylabel('Neurons')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
