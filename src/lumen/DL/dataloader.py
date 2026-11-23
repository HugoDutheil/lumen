from ast import main

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def parse_csv(csv_path, target_grid=None, wn_range=None, step=1.0):
    # Read raw CSV without headers
    raw_df = pd.read_csv(csv_path, header=None, low_memory=False)
    compound_names = raw_df.iloc[0].values
    measurement_types = raw_df.iloc[1].values
    data_values = raw_df.iloc[2:].values.astype(float)

    num_cols = data_values.shape[1]
    spectra_list = []
    metadata = []
    raw_spectra = []

    for i in range(0, num_cols-1, 2):
        name = str(compound_names[i])
        if (not name or name == "nan") and (i+1 < len(compound_names)):
            name = str(compound_names[i+1])
        measurement_type = str(measurement_types[i+1]).strip().lower()
        wn = data_values[:, i]
        intens = data_values[:, i+1]

        # Drop NaN values
        mask = ~np.isnan(wn) & ~np.isnan(intens)
        wn, intens = wn[mask], intens[mask]

        if len(wn)<10:
            continue

        # Sort by ascending wavenumber for interpolation
        sort_idx = np.argsort(wn)
        wn, intens = wn[sort_idx], intens[sort_idx]

        # Convert transmittance to absorbance : A = -log10(T)
        if "transmittance" in measurement_type:
            if np.max(intens) > 2.0:
                intens = intens / 100.0

            intens_clipped = np.clip(intens, 1e-4, 1.0)
            intens = -np.log10(intens_clipped)

        raw_spectra.append((name, measurement_type, wn, intens))
    if target_grid is None:
        if wn_range is None:
            wn_range = (max(s[2][0] for s in raw_spectra), min(s[2][-1] for s in raw_spectra))
        lo, hi = np.ceil(wn_range[0]), np.floor(wn_range[1])
        target_grid = np.arange(lo, hi+step/2, step)
    target_grid = np.asarray(target_grid, dtype=float)
    n_extrapolated = 0
    for name, measurement_type, wn, intens in raw_spectra:
        if wn[0] > target_grid[0] or wn[-1] < target_grid[-1]:
            n_extrapolated += 1

        # Interpolate to target grid
        interpolator = interp1d(wn, intens, kind='linear', bounds_error=False, fill_value="extrapolate")
        resampled_intens = interpolator(target_grid)
        spectra_list.append(resampled_intens)
        metadata.append((name, measurement_type))

    # Prepare DataFrame with target grid as columns and compound names as index
    print(f"[parse_csv] {len(spectra_list)} spectra on {target_grid[0]:.0f}-{target_grid[-1]:.0f} cm-1 "
    f"(step {target_grid[1] - target_grid[0]:g}, {len(target_grid)} points)")
    if n_extrapolated: 
        print(f"[parse_csv] warning: {n_extrapolated} spectra do not cover this range")
    X = pd.DataFrame(
    spectra_list, columns=[f"{w:.2f}" for w in target_grid], index=metadata
    )
    X.index.name = "compound"

    return X

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None


