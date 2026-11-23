import logging
import os
import re
from src import DATA
from src.lumen.PCA.PCA import interpolate_to_grid
from numba import njit
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@njit
def L2(arr1: np.ndarray, arr2: np.ndarray) -> float:
  return np.sum((arr1 - arr2) ** 2)


class set_:

    def __init__(
      self,
      initial_spectrum: np.ndarray,
      representative_name: str,
      element_name: str,
    ) -> None:
        self.representative_name: str = representative_name
        self.spectra: list[np.ndarray] = [initial_spectrum]
        self.elements: list[str] = [element_name]
        self.representative_spectrum: np.ndarray = initial_spectrum

    def add(self, spectrum: np.ndarray, element_name: str) -> None:
        self.elements.append(element_name)
        self.spectra.append(spectrum)
        self.representative_spectrum = np.mean(self.spectra, axis=0)

    @property
    def size(self) -> int:
        return len(self.elements)

    def __contains__(self, el: str) -> bool:
        return el in self.elements

    def __str__(self) -> str:
        return self.representative_name


def aggregate_spectra(
    _file: os.PathLike | None = None, epsilon: float = 1e0, plot: bool = False
) -> list[set_]:
  if _file is None:
    base_path = DATA
    datasets_path = [
        file for f in os.listdir(base_path) if os.path.isfile(file := os.path.join(base_path, f))
    ]
  else:
    datasets_path = [_file]

  grid = np.arange(np.ceil(500 / 2) * 2, np.floor(3500 / 2) * 2 + 1, 2)

  sets: list = []

  for file in datasets_path:
    data = pd.read_csv(file, dtype=str)
    for i in tqdm(range(0, data.shape[1] - 2, 2)):
      entry_name: str = str(data.axes[-1][i + 1])
      entry = data.iloc[2:, i : i + 2]
      waveNumber = np.array(entry.iloc[:, 0].astype(float))
      absorbance = np.array(entry.iloc[:, 1].astype(float))

      snapped_values = interpolate_to_grid(waveNumber, absorbance, grid)

      if not sets:
        sets.append(set_(snapped_values, entry_name, entry_name))
        continue

      best_set = None
      min_dist = float("inf")

      for _set in sets:
        dist = L2(snapped_values, _set.representative_spectrum)
        if dist < min_dist:
          min_dist = dist
          best_set = _set

      if min_dist < epsilon and best_set is not None:
        best_set.add(snapped_values, entry_name)
      else:
        sets.append(set_(snapped_values, entry_name, entry_name))

  print("Nsets = ", len(sets))
  sizes = np.empty(len(sets))
  for idx_set in range(len(sets)):
    sizes[idx_set] = sets[idx_set].size

  print("Mean size = ", np.mean(sizes))
  if plot:
    plt.figure()
    plt.hist(sizes, bins=len(sets))
    plt.xlabel(f"set size")
    plt.ylabel(f"counts")
    plt.grid(True)
    plt.savefig(
        f"set_sizes_histogram_{epsilon}.png", dpi=300, bbox_inches="tight"
    )

  return sets
