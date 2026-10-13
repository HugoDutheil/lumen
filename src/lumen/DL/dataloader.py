from ast import main

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def parse_csv(csv_path, target_grid=np.linspace(400,4000,3601)):
    # Read raw CSV without headers
    raw_df = pd.read_csv(csv_path, header=None, low_memory=False)
    compound_names = raw_df.iloc[0].values
    measurement_types = raw_df.iloc[1].values
    data_values = raw_df.iloc[2:].values.astype(float)

    num_cols = data_values.shape[1]
    spectra_list = []
    metadata = []

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

        # Interpolate to target grid
        interpolator = interp1d(wn, intens, kind='linear', bounds_error=False, fill_value="extrapolate")
        resampled_intens = interpolator(target_grid)
        spectra_list.append(resampled_intens)
        metadata.append((name, measurement_type))

    # Prepare DataFrame with target grid as columns and compound names as index

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


