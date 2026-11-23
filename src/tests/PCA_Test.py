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
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_base_name(name: str) -> str:
    try:
        base_name, _ = re.match(r'^(.*)@(\d+)$', name).groups()
        return base_name
    except:
        return name

def split_name_percentage(name_perc: str) -> tuple[str | None, int | None]:
    try:
        name, perc = re.match(r'^(.*)_(\d+)%$', name_perc).groups()
    except Exception as e:
        print(f"Error while splitting name: {e}")
        return None, None
    return name, int(perc)


def get_names_percentages(name: str) -> list[tuple[str, int]]:
    first_name_perc, second_name_perc = name.split(" + ")
    first_name, first_perc = split_name_percentage(first_name_perc)
    sec_name, sec_perc = split_name_percentage(second_name_perc)

    return [(first_name, int(first_perc)), (sec_name, int(sec_perc))]


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
    wn: np.ndarray,
    PCA_matrix: np.ndarray,
    id2sets: list[set_],
    entry_names: np.ndarray,
    n_predictions: int = 2
) -> tuple[list[int] | None, list[_set]| None]:
    

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

    file: os.PathLike = os.path.join( 
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

   
#     epsilons = np.logspace(-1, 5, 7)
    epsilons = [3e1]

    error_list = np.empty_like(epsilons)
    unprocessable_list = np.empty_like(epsilons)
    set_size_list = np.empty_like(epsilons)
    
    pc_scores = U*S
    pc1_all = pc_scores[:,0]
    pc2_all = pc_scores[:,1]

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
    
        name2idx = {name: idx for idx, name in enumerate(entry_names)}
        mispredicted_els = []
        
        with tqdm(total=num_mixtures, desc=f"Processing epsilon={eps:.1e}") as pbar:
            for i in range(0, mixed_data.shape[1]-2, 2):
                has_error = False

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
                    has_error = True

                if len(predicted_sets) > 1 and predicted_sets[1] not in true_sets:
                    logger.info(f"Predicted {predicted_sets[1]}, True value: {[str(true_set) for true_set in true_sets]}")
                    errors += 1
                    has_error = True

                if has_error:
                    # Record predicted elements
                    for pred_name in predictions:
                        if pred_name in name2idx:
                            idx = name2idx[pred_name]
                            mispredicted_els.append({
                                "PC1": pc1_all[idx],
                                "PC2": pc2_all[idx],
                                "Element": pred_name,
                                "Role": "Predicted",
                                "Mixture": mixed_entry_name,
                            })
                    # Record true elements for this mixture
                    for true_name in [first_name, sec_name]:
                        base_t = get_base_name(true_name)
                        for e_name, idx in name2idx.items():
                            if get_base_name(e_name) == base_t:
                                mispredicted_els.append({
                                    "PC1": pc1_all[idx],
                                    "PC2": pc2_all[idx],
                                    "Element": e_name,
                                    "Role": "True",
                                    "Mixture": mixed_entry_name,
                                })

                pbar.set_postfix({"unprocessable": f'{unprocessable}', "Errors": f'{errors}'})
                pbar.update(1)

        error_list[eps_idx] = errors
        unprocessable_list[eps_idx] = unprocessable
        set_size_list[eps_idx] = len(sets)
        
        fig = go.Figure()

        cluster_indices = []
        for name in entry_names:
            base_name = get_base_name(name)
            target_set = baseId2set.get(base_name)
            if target_set in sets:
                cluster_indices.append(f"Set {sets.index(target_set)}")
            else:
                cluster_indices.append("Noise / Unassigned")

        df_bg = pd.DataFrame({
            "PC1": pc1_all,
            "PC2": pc2_all,
            "Element": entry_names,
            "Cluster": cluster_indices
        })

        unique_clusters = sorted([c for c in df_bg["Cluster"].unique() if c != "Noise / Unassigned"])
        palette = px.colors.qualitative.Alphabet

        shapes = []
        for idx, cluster_name in enumerate(unique_clusters):
            cluster_data = df_bg[df_bg["Cluster"] == cluster_name]
            color = palette[idx % len(palette)]
            
            # Add scatter points for the cluster
            fig.add_trace(go.Scatter(
                x=cluster_data["PC1"],
                y=cluster_data["PC2"],
                mode='markers',
                marker=dict(size=6, color=color),
                text=cluster_data["Element"],
                name=cluster_name,
                legendgroup="clusters",
            ))
            
            points = cluster_data[["PC1", "PC2"]].values
            if len(points) >= 3:
                try:
                    # 1. Get the tight convex hull vertices
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # 2. Add padding: push vertices outward from their geometric center
                    center = np.mean(hull_points, axis=0)
                    vectors = hull_points - center
                    distances = np.linalg.norm(vectors, axis=1, keepdims=True)
                    # Expand outward by a small fixed offset (adjust 0.15 / scaling if needed)
                    padded_hull = center + vectors * (1 + 0.15 / np.maximum(distances, 1e-5))
                    
                    # 3. Smooth the polygon using a periodic B-spline
                    tck, u = splprep([padded_hull[:, 0], padded_hull[:, 1]], s=0, per=True)
                    smooth_u = np.linspace(0, 1, 200)
                    smooth_x, smooth_y = splev(smooth_u, tck)
                    
                    # 4. Build a smooth SVG path string
                    path_str = f"M {smooth_x[0]},{smooth_y[0]}"
                    for sx, sy in zip(smooth_x[1:], smooth_y[1:]):
                        path_str += f" L {sx},{sy}"
                    path_str += " Z"
                    
                    shapes.append(dict(
                        type="path",
                        path=path_str,
                        line=dict(color=color, width=2, dash="dot"),
                        fillcolor=color,
                        opacity=0.08
                    ))
                except Exception:
                    pass

        # 3. Add True vs Predicted error markers on top
        df_err = pd.DataFrame(mispredicted_els)
        if not df_err.empty:
            error_colors = px.colors.qualitative.Plotly
            for mix_idx, (mix_name, group_df) in enumerate(df_err.groupby("Mixture")):
                err_color = error_colors[mix_idx % len(error_colors)]
                
                true_group = group_df[group_df["Role"] == "True"]
                pred_group = group_df[group_df["Role"] == "Predicted"]
                
                if not true_group.empty:
                    fig.add_trace(go.Scatter(
                        x=true_group["PC1"], y=true_group["PC2"],
                        mode='markers',
                        marker=dict(size=10, color=err_color, symbol='circle', line=dict(width=1, color='black')),
                        text=true_group["Element"],
                        name=f"{mix_name} (True)",
                        legendgroup=mix_name,
                    ))
                
                if not pred_group.empty:
                    fig.add_trace(go.Scatter(
                        x=pred_group["PC1"], y=pred_group["PC2"],
                        mode='markers',
                        marker=dict(size=12, color=err_color, symbol='x', line=dict(width=2, color='black')),
                        text=pred_group["Element"],
                        name=f"{mix_name} (Pred)",
                        legendgroup=mix_name,
                    ))

        fig.update_layout(
            title=f"PC1 vs PC2 (epsilon={eps:.1e})",
            xaxis_title="PC1",
            yaxis_title="PC2",
            hovermode="closest",
            shapes=shapes,
            legend=dict(itemsizing='constant')
        )
        fig.show()

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
def nnls(A, b, epsilon=1e-3) -> np.ndarray[float]:
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
