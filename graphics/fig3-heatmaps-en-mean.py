import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Defina o tamanho da fonte
font_scale = 1.5

# Configure Seaborn com o tamanho da fonte desejado
sns.set(font_scale=font_scale)

# Dados fornecidos (convertendo para valores positivos)
rgrp_values = {
    'ALS Activity': [abs(x) for x in [-40.13, -47.68, -61.19, -64.33, -63.74, -65.57]],
    'NCF Activity': [abs(x) for x in [-20.85, -22.83, -29.31, -37.13, -34.90, -32.33]],
    'ALS Gender': [abs(x) for x in [-20.78, -29.04, -36.29, -39.91, -40.07, -40.75]],
    'NCF Gender': [abs(x) for x in [-22.19, -29.82, -34.95, -36.70, -38.63, -39.07]],
    'ALS Age': [abs(x) for x in [-9.82, -12.03, -19.37, -20.01, -21.68, -22.91]],
    'NCF Age': [abs(x) for x in [-13.34, -18.63, -22.84, -25.96, -26.65, -23.95]]
}

rmse_values = {
    'ALS Activity': [0.19, 0.08, 0.05, 0.05, 0.05, 0.06],
    'NCF Activity': [-0.22, -0.27, -0.32, -0.31, -0.32, -0.32],
    'ALS Gender': [0.81, 0.98, 1.16, 1.26, 1.27, 1.30],
    'NCF Gender': [0.15, 0.29, 0.42, 0.45, 0.52, 0.54],
    'ALS Age': [0.78, 0.92, 1.06, 1.15, 1.08, 1.11],
    'NCF Age': [0.05, 0.10, 0.22, 0.25, 0.46, 0.63]
}

# Calculando as médias para cada algoritmo e dataset
medias_rgrp = np.array([
    [
        np.mean(rgrp_values['ALS Activity']),
        np.mean(rgrp_values['ALS Gender']),
        np.mean(rgrp_values['ALS Age'])
    ],
    [
        np.mean(rgrp_values['NCF Activity']),
        np.mean(rgrp_values['NCF Gender']),
        np.mean(rgrp_values['NCF Age'])
    ]
])

medias_rmse = np.array([
    [
        np.mean(rmse_values['ALS Activity']),
        np.mean(rmse_values['ALS Gender']),
        np.mean(rmse_values['ALS Age'])
    ],
    [
        np.mean(rmse_values['NCF Activity']),
        np.mean(rmse_values['NCF Gender']),
        np.mean(rmse_values['NCF Age'])
    ]
])

# Definindo os labels para os eixos
algorithms = ['ALS', 'NCF']
datasets = ['Activity', 'Gender', 'Age']

# Criando os heatmaps
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap para R_{grp}
sns.heatmap(medias_rgrp, ax=axs[0], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=datasets, yticklabels=algorithms)
axs[0].set_title('Average Group Unfairness Reduction', fontsize=16)
axs[0].set_xlabel('Clustering', fontsize=16)
axs[0].set_ylabel('Algorithm', fontsize=16)

# Heatmap para RMSE
sns.heatmap(medias_rmse, ax=axs[1], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=datasets, yticklabels=algorithms)
axs[1].set_title('Average RMSE Increase', fontsize=16)
axs[1].set_xlabel('Clustering', fontsize=16)
axs[1].set_ylabel('Algorithm', fontsize=16)

plt.tight_layout()
plt.show()
