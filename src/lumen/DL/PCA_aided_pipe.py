import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from tqdm import tqdm

class PCAPipeline():
    def __init__(self, model, pca_dim=160, parser=None):
        self.parser = parser
        self.pca_dim = pca_dim
        self.model = model

def run_pca(pcapipe: PCAPipeline, train_data, test_data):
    pca_dim = pcapipe.pca_dim
    pca = PCA(n_components=pca_dim)
    train_d = preprocess_spectra(train_data)
    test_d = preprocess_spectra(test_data)
    train_pca = pca.fit_transform(train_d)
    test_pca = pca.transform(test_d)
    return train_pca, test_pca

def preprocess_spectra(X, window=21, polyorder=3, deriv=2):
    Xd = savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1)
    return Xd - Xd.mean(axis=1, keepdims=True)

def analyze_pca(train_pca, train_labels, labels_names=None, k=5):
    """
    Compares unsupervised PCA scores (16000, 144) to ground-truth labels (16000, 144).
    1. Alignment Analysis: Measures maximum correlation between each compound and any PC.
    2. Linear Benchmark (OLS): Solves W = (X^T X)^(-1) X^T Y in closed form to test 
       how well linear regression can recover concentrations from PCA scores.
    """
    n_samples, n_components = train_pca.shape
    n_compounds = train_labels.shape[1]
    
    print(f"--- PCA SANITY CHECK ({n_samples} samples, {n_components} components) ---")
    # Center both matrices to compute Pearson correlation across all samples
    pca_centered = train_pca - train_pca.mean(axis=0)
    labels_centered = train_labels - train_labels.mean(axis=0)

    # Standard deviations
    pca_std = np.std(train_pca, axis=0) + 1e-8
    labels_std = np.std(train_labels, axis=0) + 1e-8

    # Correlation matrix: shape (n_compounds, n_components)
    # entry [c, p] = correlation between compound c and principal component p
    corr_matrix = (labels_centered.T @ pca_centered) / (n_samples * np.outer(labels_std, pca_std))
    
    # For each compound, find its highest correlated PC
    max_corrs = np.max(np.abs(corr_matrix), axis=1)
    best_pcs = np.argmax(np.abs(corr_matrix), axis=1)
    
    print(f"Average max correlation of a compound to a single PC: {np.mean(max_corrs):.3f}")
    print(f"Compounds with strong single-PC correlation (|r| > 0.5): {np.sum(max_corrs > 0.5)} / {n_compounds}")
    print(f"Compounds with weak single-PC correlation (|r| < 0.2):   {np.sum(max_corrs < 0.2)} / {n_compounds}")
    
    # ---------------------------------------------------------
    #Closed-form Linear Regression Baseline (OLS)
    # W = (X^T X + lambda*I)^(-1) X^T Y  (Ridge/OLS closed-form)
    ridge_lambda = 1e-3
    XtX = train_pca.T @ train_pca + ridge_lambda * np.eye(n_components)
    W = np.linalg.solve(XtX, train_pca.T @ train_labels)  # shape: (n_components, n_compounds)
    
    # Linear predictions from PCA
    pred_labels = train_pca @ W
    
    # Evaluate sample 0 as a concrete sanity check
    sample_idx = 50
    true_present = np.where(train_labels[sample_idx] > 0)[0]
    
    # Top-k predicted compounds for this sample by linear PCA
    pred_sample = pred_labels[sample_idx]
    top_k_pred = np.argsort(pred_sample)[-k:][::-1]
    
    print(f"\n--- SAMPLE #{sample_idx} INSPECTION ---")
    print(f"Ground Truth Nonzero Compounds (indices): {true_present.tolist()}")
    for idx in true_present:
        name = labels_names[idx] if labels_names is not None else f"Comp_{idx}"
        print(f"  Actual: {name} | True Conc: {train_labels[sample_idx, idx]:.4f} | Linear PCA Pred: {pred_sample[idx]:.4f}")
        
    print(f"\nTop-{k} Predicted Compounds by Linear PCA:")
    for rank, idx in enumerate(top_k_pred, 1):
        name = labels_names[idx] if labels_names is not None else f"Comp_{idx}"
        is_hit = "✓ HIT" if idx in true_present else "✗ MISS"
        print(f"  {rank}. [{is_hit}] {name} (index {idx}) -> Score: {pred_sample[idx]:.4f}")

    # Overall Linear MAE across entire training set
    mae = np.mean(np.abs(pred_labels - train_labels))
    print(f"\nOverall Closed-Form Linear MAE on PCA scores: {mae:.5f}")
    print("-------------------------------------------------------")

import os
import sys
from pathlib import Path

try:
    from lumen.PCA.PCA import train as pca_pipeline_train
    from lumen.PCA.parser import Parser
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lumen.PCA.PCA import train as pca_pipeline_train
    from lumen.PCA.parser import Parser

def pca_baseline_basis(csv_path, mtx_path, grid, df_clean=None, retrain=True):
    mtx_path = str(mtx_path)
    if retrain or not os.path.exists(mtx_path):
        if os.path.exists(mtx_path):
            open(mtx_path, "w").close()               # Parser.save asks [y/n/c] when the file is not empty
        if df_clean is None:
            ok = pca_pipeline_train(data_base_path=str(csv_path), output_path=mtx_path,
                                    min_wavenumber=float(grid[0]), max_wavenumber=float(grid[-1]),
                                    step_size=float(grid[1] - grid[0]))
        else:
            U, S, Vt = np.linalg.svd(df_clean.values.astype(np.float64), full_matrices=False)
            ok = Parser().save(mtx_path, str(csv_path), U, S, Vt, df_clean.columns.astype(float).to_numpy())
        if not ok:
            raise RuntimeError(f"The PCA pipeline could not write {mtx_path}")
    title, U, S, Vt, basis_grid = Parser().parse(mtx_path)
    print(f"[PCA pipeline] basis from {title}: {Vt.shape[0]} components on "
          f"{basis_grid.squeeze()[0]:.0f}-{basis_grid.squeeze()[-1]:.0f} cm-1 ({Vt.shape[1]} points)")
    return Vt, basis_grid.squeeze()

def project_on_basis(X, wn, Vt, basis_grid, n_components):
    """Spectra on the DL grid `wn` -> scores on the first n_components of the PCA pipeline's basis."""
    X = np.asarray(X, dtype=np.float64)
    if len(basis_grid) != len(wn) or not np.allclose(basis_grid, wn):
        X = np.array([np.interp(basis_grid, wn, x) for x in X])   # resample onto the basis grid
    return (X @ Vt[:n_components].T).astype(np.float32)
