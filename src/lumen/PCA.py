import pandas as pd
import numpy as np
import os
from numba import njit
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from parser import Parser
import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@njit
def closest_int(x: float) -> int:
    if np.isnan(x):
        return 0
    tmp = np.floor(2*x) 
    tmp += tmp%2
    return tmp//2

# @njit
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

def is_csv(path: os.PathLike) -> bool:
    """
    Returns whether or not the provided file is a csv file
    """
    root, extension = os.path.splitext(path)
    return extension == ".csv"

@njit
def calculate_pca_acceled(values, grid):
    n_spectra = values.shape[1] // 2
    result = np.empty((n_spectra, grid.size), dtype=np.float64)

    for spectrum in range(n_spectra):
        wn = values[:, 2 * spectrum]
        absorbance = values[:, 2 * spectrum + 1]

        result[spectrum] = interpolate_to_grid(
            wn,
            absorbance,
            grid
        )

    return result

def calculate_PCA(file: os.PathLike, grid: np.array) -> np.ndarray:
        data = pd.read_csv(file, dtype=str)
        values = data.iloc[2:].to_numpy(dtype=np.float64)
        
        return calculate_pca_acceled(values, grid)


def train(data_base_path: os.PathLike, output_path: os.PathLike, min_wavenumber: float, max_wavenumber: float, step_size: float = np.nan, number_steps: int = np.nan) -> bool:
    """
    Trains the PCA matrix based on the data in the given directory and output the matrix in the given file 

    Arg:
        data_base_path: os.PathLike 
            if directory: path in which to retrieve the files to train on
            if file: directly uses the file

        output_path: os.PathLike
            file path to which we dump the matrix

        min_wavenumber: float
            minimum value taken by the spectrum

        max_wavenumber: float
            maximum value taken by the spectrum
        
    Optional args:
        step_size: float
            step size between min and max wavenumber, corresponds to the precision of the measurement
            Exclusivity with number_steps 

            Defaults: No default, overriden by number_steps if nothing is given

        number_steps: int
            number of steps to take between min and max wavenumber, corresponds to the precision of the measurement
            Exclusivity with step_size

            Default:  1000 steps

    Return:
        bool: 
            False if the function failed 
            True if the function succeeded

    """

    if max_wavenumber <= min_wavenumber:
        raise ValueError("Max is smaller or equal to Min")
    
    use_steps = True
    if not np.isnan(step_size):
        use_steps = False

    if not np.isnan(number_steps) and not use_steps:
        # This implies that both fields were filled
        raise ValueError("Can only process either number of steps or step size, not both")

        
    isFile: bool = False
    datasets_path = []
    if os.path.isfile(data_base_path):
        isFile = True

    if isFile and not is_csv(data_base_path):
        logger.error("Provided file is not a csv file.")
        return False

    if not isFile and not os.path.isdir(data_base_path):
        logger.error(f"No such file or directory: {data_base_path}")
        return False

    if isFile:
        datasets_path.append(data_base_path)
    else:
        for file in os.scandir(data_base_path):
            if os.path.isfile(file, follow_symlinks=False) or not is_csv(file):
                continue

            datasets_path.append(os.path.join(data_base_path, file))

    if not datasets_path:
        logger.error(f"Directory contains to usable file, expecting *.csv formated data: {data_base_path}")
        return False

    # Common grid on which I ll snap the values/interpolate for PCA
    if use_steps:
        grid = np.linspace(
            np.ceil(min_wavenumber/ 2) * 2,
            np.floor(max_wavenumber/ 2) * 2 + 1,
            number_steps
        )
    else:
        grid = np.arange(
            np.ceil(min_wavenumber/ 2) * 2,
            np.floor(max_wavenumber/ 2) * 2 + 1,
            step_size 
        )

    logger.info("Starting to read files")
    for file in datasets_path:
        PCA_matrix = calculate_PCA(file, grid)

        filter = ~np.isnan(PCA_matrix).any(axis=0)
        PCA_matrix = PCA_matrix[:, filter].astype(float)
        filtered_grid = grid[filter] 

        meanPCA = PCA_matrix.mean(axis=0)
        # Note: first I tried centering the PCA matrix by subtracting the mean, I had a vague rememberance that 
        # you had to do that, but apparently not. Not sure if I'm just misremembering or if there are specific
        # use cases. Here uncentered is much more performant.

        print(f'{PCA_matrix.shape=}')

        U, S, Vt = np.linalg.svd(PCA_matrix, full_matrices=False)
        
        parser = Parser()
        parser.save(output_path, file, U, S, Vt, filtered_grid)

    return True
