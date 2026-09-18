import pandas as pd
import numpy as np
import os
from numba import njit
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import re
from tqdm import tqdm


@njit
def closest_int(x: float) -> int:
    if np.isnan(x):
        return 0
    tmp = np.floor(2*x) 
    tmp += tmp%2
    return tmp//2

@njit
def interpolate_to_grid(wn, absorbance, grid):
    valid = ~np.isnan(wn) & ~np.isnan(absorbance)

    wn = wn[valid]
    absorbance = absorbance[valid]

    indexes = np.argsort(wn)
    wn = wn[indexes]
    absorbance = absorbance[indexes]

    result = np.full(grid.shape, np.nan, dtype=float)

    inside = (grid >= wn[0]) & (grid <= wn[-1])
    result[inside] = np.interp(grid[inside], wn, absorbance)

    return result

base_path = os.path.join(os.path.dirname(__file__), "../../data/scrambled/pure")
datasets_path = []

for file in os.listdir(base_path):
    datasets_path.append(os.path.join(base_path, file))


# Common grid on which I ll snap the values/interpolate for PCA
grid = np.arange(
    np.ceil(300/ 2) * 2,
    np.floor(3500/ 2) * 2 + 1,
    2
)
print("Starting to read files")
for m, file in enumerate(datasets_path):
    if m != 2:
        continue
    data = pd.read_csv(file, dtype=str)
    PCA_matrix = []
    entry_names = []

    print(f"Working on {file}")
    for i in tqdm(range(0, data.shape[1]-2, 2)):
        entry_names.append(data.axes[-1][i+1])
        entry = data.iloc[2:, i:i + 2]
        waveNumber = np.array(entry.iloc[:, 0].astype(float))
        absorbance = np.array(entry.iloc[:, 1].astype(float))
        
        PCA_matrix.append(interpolate_to_grid(waveNumber, absorbance, grid))

    PCA_matrix = np.array(PCA_matrix)
    filter = ~np.isnan(PCA_matrix).any(axis=0)
    PCA_matrix = PCA_matrix[:, filter].astype(float)
    filtered_grid = grid[filter] 

    meanPCA = PCA_matrix.mean(axis=0)
    # Note: first I tried centering the PCA matrix by subtracting the mean, I had a vague rememberance that 
    # you had to do that, but apparently not. Not sure if I'm just misremembering or if there are specific
    # use cases. Here uncentered is much more performant.

    print(f'{PCA_matrix.shape=}')

    U, S, Vt = np.linalg.svd(PCA_matrix, full_matrices=False)

    PC1, PC2 = Vt[0], Vt[1]
    
    eigVal = S**2/(PCA_matrix.shape[0] -1)
    var = eigVal/eigVal.sum()

    scores = U*S 
    
    mixed_file = file.replace("pure", "mixed")
    mixed_data = pd.read_csv(mixed_file, dtype=str)
    loss = 0
    unprocessable = 0
    with tqdm(total=mixed_data.shape[1]//2 - 1, desc="Processing mixtures") as pbar:
        for i in range(0, mixed_data.shape[1]-2, 2):
            mixed_entry_name = mixed_data.axes[-1][i+1]
            mixed_entry = mixed_data.iloc[2:, i:i + 2]
            mixed_waveNumber = np.array(mixed_entry.iloc[:, 0].astype(float))
            mixed_absorbance = np.array(mixed_entry.iloc[:, 1].astype(float))
    #         print("SIZES = ", mixed_waveNumber.size, mixed_absorbance.size, filtered_grid.size)
    #         print(
    #             mixed_entry_name,
    #             "total:", len(mixed_waveNumber),
    #             "NaN wn:", np.isnan(mixed_waveNumber).sum(),
    #             "NaN absorbance:", np.isnan(mixed_absorbance).sum(),
    #             "valid:", np.sum(
    #                 ~np.isnan(mixed_waveNumber) &
    #                 ~np.isnan(mixed_absorbance)
    #             )
    #         )

            mixture = interpolate_to_grid(mixed_waveNumber, mixed_absorbance, filtered_grid).astype(float)
            mixture_scores = mixture@Vt.T
            
            distances = np.linalg.norm(
                scores[:, :2] - mixture_scores[:2],
                axis=1
            )

            closest = np.argsort(distances)

            coefficients, residual_norm = nnls(
                PCA_matrix.T,
                mixture
            )
            coefficients /= coefficients.sum()
            # TODO: use a meaningful threshhold
            threshhold = 0.05 # 5%
            coefficients = np.array([coefficient if coefficient > threshhold else 0 for coefficient in coefficients])
            coefficients /= coefficients.sum()
                
            sorted_indices = np.argsort(coefficients, descending=True)

            selected_indices = []
            seen_names = set()

            for idx in sorted_indices:
                base_name = re.sub(r'_\d+$', '', entry_names[idx])

                if base_name not in seen_names:
                    seen_names.add(base_name)
                    selected_indices.append(idx)

                if len(selected_indices) == 3:
                    break

#             plt.figure()
#             plt.plot(mixed_waveNumber, mixed_absorbance, label="Mixed sample")

    #         for idx in selected_indices:
    #             base_name = re.sub(r'_\d+$', '', entry_names[idx])
    # 
    #             print(
    #                 f"{entry_names[idx]} ({base_name}): "
    #                 f"{coefficients[idx]:.4f}"
    #             )
    #         print(f"Actual composition: {mixed_entry_name}\n")
            first, second = mixed_entry_name.split(' + ')

            firstName, firstPercent = re.match(r'^(.*?)_(\d*?)%$', first).groups()
            secName, secPercent = re.match(r'^(.*?)_(\d*?)%$', second).groups()
            
            idxFirst = [idx for idx in sorted_indices if re.match(firstName, entry_names[idx])]
            idxSec = [idx for idx in sorted_indices if re.match(secName, entry_names[idx])]

            if idxFirst and idxSec:
                foundFirstPercent = coefficients[idxFirst[0]]*100
                foundSecPercent = coefficients[idxSec[0]]*100

                loss += ((int(firstPercent) - foundFirstPercent)**2 + (int(secPercent) - foundSecPercent)**2)**0.5
            else: 
                unprocessable += 1

            pbar.set_postfix({"loss" : f'{loss:.2f}', "unprocessable":f'{unprocessable}'})
            pbar.update(1)
        print(f'Total loss of file: {loss}') 
    #             plt.plot(
    #                 np.array(data.iloc[2:, idx * 2].astype(float)),
    #                 np.array(data.iloc[2:, idx * 2 + 1].astype(float)),
    #                 linestyle='--',
    #                 label=f"{base_name} ({coefficients[idx]:.4f})"
    #             )


    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
