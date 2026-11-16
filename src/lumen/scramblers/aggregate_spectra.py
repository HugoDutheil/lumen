import pandas as pd
import numpy as np
import os
from src.lumen.PCA.PCA import interpolate_to_grid
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
from src import DATA


@njit
def L2(arr1: np.array, arr2: np.array) -> float:
    return np.sum((arr1-arr2)**2)

class set_:
    def __init__(self, representative_spectrum: np.array = [], representative_name: str = "", elements: list = []) -> None:
        self.representative_name: str = representative_name
        self.representative_spectrum: np.array = representative_spectrum
        self.elements: list = [elements]

    def add(self, element: os.PathLike) -> None:
        self.elements.append(element)

    @property 
    def size(self) -> int:
        return len(self.elements)
    
    def __contains__(self, el: str) -> bool:
        return el in self.elements
    
    def __str__(self) -> str:
        return self.representative_name


def aggregate_spectra(_file: str | None = None, epsilon: float = 1e0) -> list[set_]:
    
    if _file is None:
        base_path = DATA
        datasets_path = [
            file
            for f in os.listdir(base_path)
            if os.path.isfile(
                file := os.path.join(base_path, f)
            )
        ]
    else:
        datasets_path = [_file]

    grid = np.arange(
        np.ceil(500/ 2) * 2,
        np.floor(3500/ 2) * 2 + 1,
        2
    )

    sets: list = []

    for file in datasets_path:
        data = pd.read_csv(file, dtype=str)
        for i in tqdm(range(0, data.shape[1]-2, 2)):
            entry_name = data.axes[-1][i+1]
            entry = data.iloc[2:, i:i + 2]
            waveNumber = np.array(entry.iloc[:, 0].astype(float))
            absorbance = np.array(entry.iloc[:, 1].astype(float))

            snapped_values = interpolate_to_grid(waveNumber, absorbance, grid)
            
            if i == 0:
                sets.append(set_(snapped_values, entry_name, entry_name))
                continue
            
            inserted: bool = False
            for _set in sets:
                if L2(snapped_values, _set.representative_spectrum) < epsilon:
#                     print("Added element to set", i//2)
                    _set.add(entry_name)
                    inserted = True
                    break

            if not inserted:
                sets.append(set_(snapped_values, entry_name, entry_name))
#                 print("created new set", i//2)

    print("Nsets = ", len(sets))
    sizes = np.empty_like(sets)
    for idx_set in range(len(sets)):
        sizes[idx_set] = sets[idx_set].size

    print("Mean size = ", np.mean(sizes))

    plt.figure()
    plt.hist(sizes, bins=len(sets))
    plt.xlabel(f'set size')
    plt.ylabel(f'counts')
    plt.grid(True)
    plt.savefig(f"set_sizes_histogram_{epsilon}.png", dpi=300, bbox_inches="tight")

    return sets
