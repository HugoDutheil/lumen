import os
import numpy as np
from src.lumen.PCA.parser import Parser
import matplotlib.pyplot as plt
from src import DATA

parser: Parser = Parser()
training_file, U, S, Vt, filtered_grid = parser.parse(
    os.path.join(DATA, "test_matrix.mtx")
)

scores = U * S 
explained_var = (S ** 2) / np.sum(S ** 2)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

num_components = len(S)
axes[0].plot(range(1, num_components + 1), explained_var * 100, marker='o', linestyle='-', color='b')
axes[0].set_title('Scree Plot')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].grid(True)

axes[1].scatter(scores[:, 0], scores[:, 1], alpha=0.7, edgecolors='k', color='teal')
axes[1].set_title('scores')
axes[1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
axes[1].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
axes[1].grid(True)

axes[2].scatter(scores[:, 0], scores[:, 1], alpha=0.4, color='gray', label='Samples')
axes[2].set_title('Biplot')
axes[2].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
axes[2].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
axes[2].grid(True)

k_comp = 3
percentages = explained_var[:k_comp] * 100
other = 100 - np.sum(percentages)
percentages = np.concat((percentages, [other]), axis=0)
labels = [str(i+1) for i in range(k_comp)]
labels = np.concat((labels, ["other"]), axis=0)

axes[3].set_title("Explained variance per component")
axes[3].pie(percentages, labels=labels, autopct='%1.1f%%')

loadings_pc1 = Vt[0]
loadings_pc2 = Vt[1]

scale_factor = np.max(np.abs(scores[:, :2])) / np.max(np.abs(Vt[:2, :]))

num_features_to_plot = min(6, Vt.shape[1])
for i in range(num_features_to_plot):
    axes[2].arrow(0, 0, loadings_pc1[i] * scale_factor, loadings_pc2[i] * scale_factor,
                  color='red', alpha=0.8, head_width=scale_factor * 0.003)
    axes[2].text(loadings_pc1[i] * scale_factor * 1.15, loadings_pc2[i] * scale_factor * 1.15,
                 f'F{i+1}', color='red', fontsize=9, weight='bold')


plt.tight_layout()
plt.savefig('pca_plots.png', dpi=300)
