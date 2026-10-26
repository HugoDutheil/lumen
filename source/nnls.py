
"""
Physics baseline: non-negative least squares (NNLS) against the reference library.
 
Each measured spectrum is modelled as
    spectrum = sum_i w_i * reference_i  +  (a + b*x + c*x^2)  +  noise,   w_i >= 0
which is exactly how generator.py builds the mixtures. A compound is declared present
when its coefficient is large *compared with its own uncertainty* (z-score), not when the
raw coefficient passes a fixed cutoff.
"""
import numpy as np
from scipy.optimize import nnls
from joblib import Parallel, delayed
 
 
def build_design(library, wavenumbers, sum_to_one=True, penalty=1e3):
    """
    library     : (n_compounds, n_points) pure spectra
    wavenumbers : (n_points,)
    Returns (A, A_fit, n_compounds):
      A     : compounds + baseline columns, used for the noise / uncertainty estimate
      A_fit : A plus one extra row pushing the compound weights to sum to 1 (if sum_to_one)
    """
    library = np.asarray(library, dtype=np.float64)
    wn = np.asarray(wavenumbers, dtype=np.float64)
    x = 2 * (wn - wn.min()) / (wn.max() - wn.min()) - 1
    B = np.stack([np.ones_like(x), x, x**2], axis=1)
    A = np.hstack([library.T, B, -B])
    n_compounds = library.shape[0]
    if sum_to_one:
        row = np.r_[np.ones(n_compounds), np.zeros(A.shape[1] - n_compounds)] * penalty
        A_fit = np.vstack([A, row])
    else:
        A_fit = A
    return A, A_fit, n_compounds
 
 
def nnls_fit(spectrum, A, A_fit, n_compounds, penalty=1e3):
    """Fit one spectrum. Returns (coefficients, z-scores), both of length n_compounds."""
    d = np.asarray(spectrum, dtype=np.float64)
    target = np.r_[d, penalty] if A_fit.shape[0] > A.shape[0] else d
    coef, _ = nnls(A_fit, target, maxiter=2000)
 
    active = np.nonzero(coef > 0)[0]
    resid = d - A @ coef
    dof = max(len(d) - len(active), 1)
    sigma2 = resid @ resid / dof                       # noise level estimated from the residual
    z = np.zeros_like(coef)
    if len(active):
        cov = np.linalg.pinv(A[:, active].T @ A[:, active]) * sigma2
        z[active] = coef[active] / np.sqrt(np.maximum(np.diag(cov), 1e-30))
    return coef[:n_compounds], z[:n_compounds]
 
 
def nnls_batch(X, library, wavenumbers, sum_to_one=True, n_jobs=-1):
    """Fit many spectra (in parallel). Returns (C, Z), each of shape (n_samples, n_compounds)."""
    A, A_fit, n = build_design(library, wavenumbers, sum_to_one)
    out = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(nnls_fit)(s, A, A_fit, n) for s in np.asarray(X)
    )
    C = np.array([c for c, _ in out], dtype=np.float32)
    Z = np.array([z for _, z in out], dtype=np.float32)
    return C, Z
 
 
def tune_threshold(scores, Y, grid=None):
    """Pick the threshold that minimizes (missed + hallucinated) on a VALIDATION set."""
    present = Y > 0
    if grid is None:
        grid = np.quantile(scores[scores > 0], np.linspace(0, 0.999, 400))
    errors = [((present & ~(scores > t)).sum() + (~present & (scores > t)).sum()) for t in grid]
    return float(grid[int(np.argmin(errors))])
 
 
def predict_nnls(C, Z, z_threshold):
    """Presence = z-score above threshold; concentrations = coefficients renormalized over detected compounds."""
    detected = Z > z_threshold
    conc = np.where(detected, C, 0.0)
    sums = conc.sum(axis=1, keepdims=True)
    conc = np.divide(conc, sums, out=np.zeros_like(conc), where=sums > 0)
    return detected, conc
 
 
def report(detected, conc, Y, title="RESULTS"):
    """Same numbers as test_mlp, so the two pipelines can be compared directly."""
    present = Y > 0
    missed = present & ~detected
    halluc = ~present & detected
    n = len(Y)
    print(f"=== {title} ===")
    print(f"Samples: {n} | present components: {present.sum()}")
    print(f"Missed (FN):       {missed.sum()}  ({missed.sum()/n:.2f} per sample)")
    print(f"Hallucinated (FP): {halluc.sum()}  ({halluc.sum()/n:.2f} per sample)")
    print(f"True positives:    {(present & detected).sum()}  (recall {(present & detected).sum()/present.sum():.3f})")
    print(f"Perfectly identified samples: {((missed.sum(1) == 0) & (halluc.sum(1) == 0)).mean():.1%}")
    print(f"Present-only concentration MAE: {np.abs(conc[present] - Y[present]).mean():.4f}")
    print(f"Top-1 match: {(conc.argmax(1) == Y.argmax(1)).mean():.1%}")
    for lo, hi in [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 1.0)]:
        m = present & (Y > lo) & (Y <= hi)
        if m.any():
            print(f"  proportion {lo:.0%}-{hi:.0%}: recall {detected[m].mean():.1%}  ({m.sum()} components)")