from src.lumen.PCA.PCA import train, interpolate_to_grid, closest_int
from tqdm import tqdm
from src.lumen.PCA.parser import Parser
from src.lumen.scramblers.aggregate_spectra import aggregate_spectra, set_
from numba import njit
import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import DATA

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_base_name(name: str) -> str:
    try:
        base_name, iteration = re.match(r'^(.*)@(\d+)$', name).groups()
        return base_name
    except:
        return name

def split_name_percentage(name_perc: str) -> tuple(str, int):
    try:
        name, perc = re.match(r'^(.*)_(\d+)%$', name_perc).groups()
    except Exception as e:
        print(f"Error while splitting name: {e}")
        return None, None
    return name, perc


def get_names_percentages(name: str) -> list(tuple(str, int)):
    first_name_perc, second_name_perc = name.split(" + ")
    first_name, first_perc = split_name_percentage(first_name_perc)
    sec_name, sec_perc = split_name_percentage(second_name_perc)

    return [(first_name, first_perc), (sec_name, sec_perc)]


def map_ids_to_sets(ids: str, baseId2set: dict) -> dict:
    id2sets = {}
    for id in ids:
        base_name = get_base_name(id)
        id2sets[id] = baseId2set.get(base_name)

    return id2sets

def map_baseIds_to_sets(sets) -> dict:
    baseId2set = {}
    for _set in sets:
        for el in _set.elements:
            baseId2set[get_base_name(el)] = _set
    return baseId2set

def predict_mixture(
    wn: np.array,
    PCA_matrix: np.ndarray,
    id2sets: list[set_],
    entry_names: np.array,
    n_predictions: int = 2
) -> tuple(list[int] | None, list[_set]| None):
    

    try:
        coefficients = nnls(PCA_matrix.T, wn)
    except:
        return None, None

    coefficients /= np.sum(coefficients) 
    threshold = 0.05 # 5%

    coefficients = np.array([coef if (coef > threshold) else 0. for coef in coefficients])
    coefficients /= np.sum(coefficients)
    
    sorted_indices = np.argsort(coefficients)[::-1]
    
    selected_sets = set()
    predictions = []

    for idx in sorted_indices:
        name = entry_names[idx]
        _set = id2sets.get(name)

        if _set not in selected_sets:
            logger.debug(f'Selected index: {name}, with set: {_set}')
            selected_sets.add(_set)
            predictions.append(name)

        if len(predictions) >= n_predictions:
            break
    
    return predictions, list(selected_sets)
    
    

def process_mixtures(do_training: bool = False, print_errors: bool = False) -> None:

    if print_errors:
        logging.basicConfig(level=logging.INFO)

    file = os.path.join( 
                DATA,
                "scrambled/pure/Primpke_FTIR_Microplastics_converted_scrambled_pure.csv"
            )
    if do_training:
        succeeded = train(
            data_base_path = file,
            output_path = os.path.join(
                DATA,
                "test_matrix.mtx"
            ),
            min_wavenumber = 400,
            max_wavenumber = 4000,
            step_size=2.
        )
        if not succeeded:
            logger.error("Was unable to train")
            return

    parser = Parser()
    training_file, U, S, Vt, filtered_grid = parser.parse(os.path.join(DATA, "test_matrix.mtx"))

    PCA_matrix = (U*S)@Vt
    np.nan_to_num(PCA_matrix, nan=0., copy=False)

   
    epsilons = np.logspace(-1, 5, 7)

    error_list = np.empty_like(epsilons)
    unprocessable_list = np.empty_like(epsilons)
    set_size_list = np.empty_like(epsilons)

    for eps_idx, eps in enumerate(epsilons):
        errors = 0
        unprocessable = 0

        sets = aggregate_spectra(file, epsilon=eps)
        baseId2set = map_baseIds_to_sets(sets)

        mixed_file = file.replace("pure", "mixed")
        mixed_data = pd.read_csv(mixed_file, dtype=str)

        entry_names = pd.read_csv(training_file, dtype=str, header=None, nrows = 1).iloc[0].to_numpy()
        entry_names = entry_names[pd.notna(entry_names)]

        num_mixtures = mixed_data.shape[1] // 2 - 1
        
        id2sets = map_ids_to_sets(entry_names, baseId2set)
        
        with tqdm(total=num_mixtures, desc=f"Processing epsilon={eps:.1e}") as pbar:
            for i in range(0, mixed_data.shape[1]-2, 2):
                mixed_entry_name = mixed_data.axes[-1][i+1]
                mixed_entry = mixed_data.iloc[2:, i:i + 2]
                mixed_wavenumber = np.array(mixed_entry.iloc[:, 0].astype(float))
                mixed_absorbance = np.array(mixed_entry.iloc[:, 1].astype(float))
                wn_snapped = interpolate_to_grid(mixed_wavenumber, mixed_absorbance, filtered_grid).astype(float)
                
                true_names_percentages = get_names_percentages(mixed_entry_name)
                first_name, first_perc = true_names_percentages[0]
                sec_name, sec_perc = true_names_percentages[1]

                true_sets = {
                    baseId2set.get(get_base_name(first_name)) or id2sets.get(first_name),
                    baseId2set.get(get_base_name(sec_name)) or id2sets.get(sec_name)
                }

                np.nan_to_num(wn_snapped, nan=0., copy=False)
                
                predictions, predicted_sets = predict_mixture(wn_snapped, PCA_matrix, id2sets, entry_names, n_predictions=len(true_sets))
                logger.debug(f'{predictions}, {predicted_sets}')
                
                if predicted_sets is None or predictions is None:
                    logger.error("Something went wrong in prediction")
                    unprocessable += 1
                    continue

                if predicted_sets[0] not in true_sets:
                    logger.info(f"Predicted {predicted_sets[0]}, True value: {[str(true_set) for true_set in true_sets]}")
                    errors += 1

                if len(predicted_sets) > 1 and predicted_sets[1] not in true_sets:
                    logger.info(f"Predicted {predicted_sets[1]}, True value: {[str(true_set) for true_set in true_sets]}")
                    errors += 1

                pbar.set_postfix({"unprocessable": f'{unprocessable}', "Errors": f'{errors}'})
                pbar.update(1)

        error_list[eps_idx] = errors
        unprocessable_list[eps_idx] = unprocessable
        set_size_list[eps_idx] = len(sets)

    plt.figure()
    plt.scatter(epsilons, error_list, marker='x')
    plt.scatter(epsilons, unprocessable_list, marker='s', label='Unprocessable')
    plt.scatter(epsilons, set_size_list, marker='3', label='Number of sets')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(f'Error')
    plt.semilogx()
    plt.grid(True)
    plt.legend()
    plt.savefig('errors_epsilons.jpeg')
    return


@njit
def nnls(A, b, epsilon=1e-3):
    m, n = A.shape
    if b.size != m:
        print("Size mismatch", b.size, n, m)
        raise ValueError("Dimensions mismatch")
    
    P = np.zeros(n, dtype=np.float64)
    R = np.ones(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    w = A.T @ (b - A @ x)
    
    while not np.all(R == -1) and np.max(w) > epsilon:
        j = np.argmax(w)
        R[j] = -1
        P[j] = 1
        
        A_p = A[:, P == 1]
        if A_p.shape[0] == 0:
            break
        A_p_T = A_p.T
        
        try:
            s_p = np.linalg.inv(A_p_T @ A_p) @ A_p_T @ b
        except:
            print("Counld not calculate inverse")
            break
            
        s = np.zeros(n, dtype=np.float64)
        p_indices = np.where(P == 1)[0]

        for idx_i, j_idx in enumerate(p_indices):
            s[j_idx] = s_p[idx_i]

        while np.min(s_p) <= 0.0:
            alpha = 1.0
            for k in range(n):
                if P[k] == 1 and s[k] <= 0:
                    a_k = x[k] / (x[k] - s[k])
                    if a_k < alpha:
                        alpha = a_k
            x += alpha * (s - x)
            for j in range(n):
                if P[j] == 1 and x[j] <= 0:
                    R[j] = 1
                    P[j] = 0
            
            p_indices = np.where(P == 1)[0]
            if len(p_indices) == 0:
                print("Finished the loop")
                break
            A_p = A[:, P == 1]
            A_p_T = A_p.T
            try:
                s_p = np.linalg.inv(A_p_T @ A_p) @ A_p_T @ b
            except:
                print("Counld not calculate inverse")
                break

            s = np.zeros(n, dtype=np.float64)
            for idx_i, j_idx in enumerate(p_indices):
                s[j_idx] = s_p[idx_i]
                
        x = s
        w = A.T @ (b - A @ x)
    return x


if __name__ == '__main__':
    process_mixtures(False, True)
