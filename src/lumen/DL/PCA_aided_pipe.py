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


