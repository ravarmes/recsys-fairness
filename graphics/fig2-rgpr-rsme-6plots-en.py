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

fig, axs = plt.subplots(3, 2, figsize=(15, 13))  # Ajustando o tamanho da figura
fig.suptitle('Rgrp and RMSE Comparison by 95-5 Strategy, Algorithm, and Clustering')

# Iterando através dos dados para criar os gráficos
for i, (title, data) in enumerate(data.items()):
    ax = axs.flat[i]
    ax.plot(data['h'], data['Rgrp'], marker='o', linestyle='-', color='tab:blue', label='Rgrp')
    ax.plot(data['h'], data['RMSE'], marker='s', linestyle='--', color='tab:orange', label='RMSE')

    # Colocando o título na parte de baixo do gráfico
    ax.text(0.5, 0.78, title, ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Definindo os ticks do eixo x para os valores inteiros especificados
    ax.set_xticks(data['h'])

    # Ocultando os rótulos do eixo x para os primeiros quatro gráficos
    if i < 4:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax.set_xlabel('Number of Matrices (h)')
    
    # Definindo o rótulo do eixo y apenas para o subplot 4
    if i == 2:
        ax.set_ylabel('Percentage Change (%)')
    ax.grid(True)
    ax.legend(loc='lower left')

    # Adicionando o ponto (0, 0) se necessário
    ax.scatter(0, 0, color='red')  # Ponto vermelho em (0, 0)

    # Ajustando as escalas do eixo y e x
    ax.set_ylim([-80, 10])  # Escala do eixo y de -100 a 10
    ax.set_xlim([min(data['h']) - 1, max(data['h']) + 1])  # Espaço antes do menor valor de h e após o maior
    ax.set_yticks(np.arange(-80, 11, 10))  # De -100 a 10, em passos de 10

# Ajustando o espaçamento entre os subplots
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.8, wspace=0.5)

# Ajustes finais para melhorar o layout e a visualização
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
