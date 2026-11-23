from src.lumen.PCA.PCA import train, interpolate_to_grid, closest_int
from tqdm import tqdm
from src.lumen.PCA.parser import Parser
from src.lumen.scramblers.aggregate_spectra import aggregate_spectra, set_
import pandas as pd
import numpy as np
import re
import os
import logging
import matplotlib.pyplot as plt
from numba import njit
from src import DATA

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@njit(nopython=True)
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

@njit(nopython=True)
def predict_mixture(mixture, PCA_matrix, entry_to_set_id):
    """JIT-compiled prediction logic using integer lookup arrays instead of dictionaries"""
    try:
        coefficients = nnls(PCA_matrix, mixture)
    except:
        return np.empty(0, dtype=np.int64), True

    coef_sum = np.sum(coefficients)
    if coef_sum > 0:
        coefficients /= coef_sum
        
    threshold = 0.05
    for i in range(len(coefficients)):
        if coefficients[i] <= threshold:
            coefficients[i] = 0.0
            
    coef_sum = np.sum(coefficients)
    if coef_sum > 0:
        coefficients /= coef_sum

    sorted_indices = np.argsort(coefficients)[::-1]
    
    selected_indices = []
    seen_sets = set()
    
    for idx in sorted_indices:
        set_id = entry_to_set_id[idx]
        if set_id != -1 and set_id not in seen_sets:
            seen_sets.add(set_id)
            selected_indices.append(idx)
        if len(selected_indices) == 2:
            break
            
    return np.array(selected_indices, dtype=np.int64), False


def test_on_mixtures(do_training: bool = True, print_errors: bool = False) -> None:
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
    print(U.shape, S.shape, Vt.shape, filtered_grid.shape)
    S = S.squeeze()
    filtered_grid = filtered_grid.squeeze()
    
    PCA_matrix = (U*S)@Vt
    np.nan_to_num(PCA_matrix, nan=0., copy=False)
      
    mixed_file = file.replace("pure", "mixed")
    mixed_data = pd.read_csv(mixed_file, dtype=str)

    entry_names = pd.read_csv(training_file, dtype=str, header=None, nrows = 1).iloc[0].to_numpy()
    entry_names = entry_names[pd.notna(entry_names)]
    
    epsilons = [3] # Found to be about optimal
    errors_per_eps = np.empty_like(epsilons, dtype=float)
    num_mixtures = mixed_data.shape[1] // 2 - 1

    for eps_idx, eps in enumerate(epsilons):    
        sets = aggregate_spectra(file, eps)
        
        set_to_id = {s: i for i, s in enumerate(sets)}
        
        entry_to_set_id = np.empty(len(entry_names), dtype=np.int64)
        entry_to_set_obj = {}
        for s in sets:
            for el in s.elements:
                entry_to_set_obj[el] = s

        for idx, entry_name in enumerate(entry_names):
            target_set = entry_to_set_obj.get(entry_name)
            entry_to_set_id[idx] = set_to_id[target_set] if target_set in set_to_id else -1

        errors = 0
        loss = 0
        unprocessable = 0
        
        with tqdm(total=num_mixtures, desc=f"Processing eps={eps:.1e}") as pbar:
            for i in range(0, mixed_data.shape[1]-2, 2):
                mixed_entry_name = mixed_data.axes[-1][i+1]
                mixed_entry = mixed_data.iloc[2:, i:i + 2]
                mixed_waveNumber = np.array(mixed_entry.iloc[:, 0].astype(float))
                mixed_absorbance = np.array(mixed_entry.iloc[:, 1].astype(float))
                 
                mixture = interpolate_to_grid(mixed_waveNumber, mixed_absorbance, filtered_grid).astype(float)
                np.nan_to_num(mixture, nan=0., copy=False)
                
                selected_indices, unprocessed = predict_mixture(mixture, PCA_matrix.T, entry_to_set_id)
                
                if unprocessed or len(selected_indices) < 2:
                    unprocessable += 1
                    print("Unprocessable")
                    pbar.set_postfix({"unprocessable": f'{unprocessable}', "Errors": f'{errors}'})
                    pbar.update(1)
                    continue

                try:
                    first, second = mixed_entry_name.split(' + ')
                    firstName, firstPercent = re.match(r'^(.*)_(\d+)%$', first).groups()
                    secName, secPercent = re.match(r'^(.*)_(\d+)%$', second).groups()
                except:
                    print("Couldn't predict two values")
                    unprocessable += 1
                    pbar.set_postfix({"unprocessable": f'{unprocessable}', "Errors": f'{errors}'})
                    pbar.update(1)
                    continue

                idxFirst = [idx for idx, name in enumerate(entry_names) if re.match(firstName, name)]
                idxSec = [idx for idx, name in enumerate(entry_names) if re.match(secName, name)]

                if idxFirst and idxSec:
                    true_sets_1 = {entry_to_set_obj[entry_names[idx]] for idx in idxFirst if entry_names[idx] in entry_to_set_obj}
                    true_sets_2 = {entry_to_set_obj[entry_names[idx]] for idx in idxSec if entry_names[idx] in entry_to_set_obj}
                    all_true_sets = true_sets_1.union(true_sets_2)

                    pred_set_1 = entry_to_set_obj.get(entry_names[selected_indices[0]])
                    pred_set_2 = entry_to_set_obj.get(entry_names[selected_indices[1]])

                    is_pred_1_correct = pred_set_1 in all_true_sets if pred_set_1 else False
                    is_pred_2_correct = pred_set_2 in all_true_sets if pred_set_2 else False

                    errors += int(not is_pred_1_correct) + int(not is_pred_2_correct)
                    if not is_pred_1_correct and print_errors:
                        print(f'\nFirst prediction: {entry_names[selected_indices[0]]} of set {pred_set_1}')
                        print(f'"True Values: {all_true_sets}')

                    if not is_pred_2_correct and print_errors:
                        print(f'\nFirst prediction: {entry_names[selected_indices[1]]} of set {pred_set_2}')
                        print(f'"True Values: {[_set.representative_name for _set in all_true_sets]}')
                else:
                    unprocessable += 1
                    print("Unprocessable")
                pbar.set_postfix({"unprocessable": f'{unprocessable}', "Errors": f'{errors}'})
                pbar.update(1)
                
        errors_per_eps[eps_idx] = 100 * errors / (2 * num_mixtures)
        print("Errors: ", errors)
        print("Unprocessable: ", unprocessable)

    plt.figure()
    plt.scatter(epsilons, errors_per_eps, marker='s')
    plt.xscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'Percentage of errors')
    plt.grid(True)
    plt.savefig("error_plot.png", dpi=300, bbox_inches="tight")

test_on_mixtures(do_training=False, print_errors=True)
