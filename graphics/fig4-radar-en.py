import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = {
    'ALS Activity': {'h': [0, 3, 5, 10, 15, 20, 25], 'Rgrp': [0, -40.13, -47.68, -61.19, -64.33, -63.74, -65.57], 'RMSE': [0, 0.19, 0.08, 0.05, 0.05, 0.05, 0.06]},
    'NCF Activity': {'h': [0, 3, 5, 10, 15, 20, 25], 'Rgrp': [0, -20.85, -22.83, -29.31, -37.13, -34.90, -32.33], 'RMSE': [0, -0.22, -0.27, -0.32, -0.31, -0.32, -0.32]},
    'ALS Gender': {'h': [0, 3, 5, 10, 15, 20, 25], 'Rgrp': [0, -20.78, -29.04, -36.29, -39.91, -40.07, -40.75], 'RMSE': [0, 0.81, 0.98, 1.16, 1.26, 1.27, 1.30]},
    'NCF Gender': {'h': [0, 3, 5, 10, 15, 20, 25], 'Rgrp': [0, -22.19, -29.82, -34.95, -36.70, -38.63, -39.07], 'RMSE': [0, 0.15, 0.29, 0.42, 0.45, 0.52, 0.54]},
    'ALS Age': {'h': [0, 3, 5, 10, 15, 20, 25], 'Rgrp': [0, -9.82, -12.03, -19.37, -20.01, -21.68, -22.91], 'RMSE': [0, 0.78, 0.92, 1.06, 1.15, 1.08, 1.11]},
    'NCF Age': {'h': [0, 3, 5, 10, 15, 20, 25], 'Rgrp': [0, -13.34, -18.63, -22.84, -25.96, -26.65, -23.95], 'RMSE': [0, 0.05, 0.10, 0.22, 0.25, 0.46, 0.63]}
}

# Categorias e algoritmos
categories = ['Activity', 'Gender', 'Age']
algorithms = ['ALS', 'NCF']

# Processando dados para encontrar os máximos valores de redução de Rgrp (menores absolutos) e aumentos de RMSE (maiores valores)
data_group_injustice = [
    [max(data['ALS Activity']['Rgrp'][1:], key=abs), max(data['ALS Gender']['Rgrp'][1:], key=abs), max(data['ALS Age']['Rgrp'][1:], key=abs)],  # ALS (convertendo para absolutos)
    [max(data['NCF Activity']['Rgrp'][1:], key=abs), max(data['NCF Gender']['Rgrp'][1:], key=abs), max(data['NCF Age']['Rgrp'][1:], key=abs)]   # NCF (convertendo para absolutos)
]

data_rmse_increase = [
    [max(data['ALS Activity']['RMSE'][1:]), max(data['ALS Gender']['RMSE'][1:]), max(data['ALS Age']['RMSE'][1:])],  # ALS
    [max(data['NCF Activity']['RMSE'][1:]), max(data['NCF Gender']['RMSE'][1:]), max(data['NCF Age']['RMSE'][1:])]   # NCF
]

# Convertendo valores de Rgrp para absolutos
data_group_injustice = [[abs(value) for value in sublist] for sublist in data_group_injustice]

# Angles for radar charts
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

# First radar chart: Algorithms by Datasets for Group Injustice Reduction
for data, algorithm in zip(data_group_injustice, algorithms):
    axs[0].plot(angles, data + data[:1], label=algorithm)
    axs[0].fill(angles, data + data[:1], alpha=0.1)
axs[0].set_xticks(angles[:-1])
axs[0].set_xticklabels(categories, fontsize=12)
axs[0].set_title('Maximum Group Unfairness Reduction by Algorithms', fontsize=16)
axs[0].legend()

# Second radar chart: Algorithms by Datasets for RMSE Increase
for data, algorithm in zip(data_rmse_increase, algorithms):
    axs[1].plot(angles, data + data[:1], label=algorithm)
    axs[1].fill(angles, data + data[:1], alpha=0.1)
axs[1].set_xticks(angles[:-1])
axs[1].set_xticklabels(categories, fontsize=12)
axs[1].set_title('Maximum RMSE Increase by Algorithms', fontsize=16)
axs[1].legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
