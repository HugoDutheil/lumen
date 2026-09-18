import pandas as pd
import numpy as np
import os
from tqdm import tqdm


nMixes = 1
nScrambles = 10

WN_NOISE = 1 / 1000
ABS_NOISE = 1 / 10


def scramble_spectrum(wn, absorbance, rng):
    """
    scrambles with a gaussian envelope the wave number and absorbance given using the rng seed
    """
    scrambled_wn = rng.normal(loc=wn, scale=np.abs(wn) * WN_NOISE)
    scrambled_abs = rng.normal(loc=absorbance, scale=np.abs(absorbance) * ABS_NOISE)

    return scrambled_wn, scrambled_abs


def clean_spectrum(wn, absorbance):
    """
    Removes invalid values such as NaN or undefined values 
    """
    valid = np.isfinite(wn) & np.isfinite(absorbance)

    wn = wn[valid]
    absorbance = absorbance[valid]

    indexes = np.argsort(wn)

    return wn[indexes], absorbance[indexes]


def mix_spectra(wn1, abs1, wn2, abs2, percentage, rng):
    """
    Mixes two spectra together using the provided percentage and adds a gaussian noise to it 
    """
    out_wn = []
    out_abs = []

    for wn, absval in zip(wn1, abs1):

        j = np.argmin(np.abs(wn2 - wn))

        source_wn = wn2[j]
        source_abs = abs2[j]

        mixed_wn = (percentage * wn + (1 - percentage) * source_wn)

        mixed_abs = (percentage * absval + (1 - percentage) * source_abs)

        mixed_wn = rng.normal(mixed_wn, abs(mixed_wn) * WN_NOISE)

        mixed_abs = rng.normal(mixed_abs, abs(mixed_abs) * ABS_NOISE)

        out_wn.append(mixed_wn)
        out_abs.append(mixed_abs)

    return np.asarray(out_wn), np.asarray(out_abs)


def pad_columns(columns):
    """
    Formats the columns as to all have the same size so the pandas DataFrame doesn't die on me
    """
    max_length = max(len(column) for column in columns)

    padded = []

    for column in columns:
        padded.append(
            np.pad(
                column,
                (0, max_length - len(column)),
                mode="constant",
                constant_values=np.nan
            )
        )

    return np.column_stack(padded)


base_path = os.path.join(
    os.path.dirname(__file__),
    "../../data"
)

datasets_path = [
    file
    for f in os.listdir(base_path)
    if os.path.isfile(
        file := os.path.join(base_path, f)
    )
]

rng = np.random.default_rng()

for file in datasets_path:
    data = pd.read_csv(file, dtype=str)

    print(f"Working on {file}")
    pure_columns = []
    pure_headers = []
    pure_subheaders = []

    mixed_columns = []
    mixed_headers = []
    mixed_subheaders = []

    original_spectra = []
    entry_names = []

    for i in tqdm(
        range(0, data.shape[1] - 2, 2),
        desc="Reading spectra"
    ):

        entry_name = data.columns[i + 1]

        entry = data.iloc[2:, i:i + 2]

        wn = entry.iloc[:, 0].astype(float).to_numpy()
        absorbance = entry.iloc[:, 1].astype(float).to_numpy()

        wn, absorbance = clean_spectrum(wn, absorbance)

        original_spectra.append((wn, absorbance))
        entry_names.append(entry_name)


    for entry_idx, (entry_name, (wn, absorbance)) in enumerate(zip(entry_names, original_spectra)):
        for scramble_idx in range(nScrambles):

            scrambled_wn, scrambled_abs = scramble_spectrum(wn, absorbance, rng)

            pure_headers.extend([
                "",
                f"{entry_name}_{scramble_idx + 1}"
            ])

            pure_subheaders.extend([
                "Wavenumber [1/cm]",
                "Absorbance"
            ])

            pure_columns.extend([
                scrambled_wn,
                scrambled_abs
            ])

    for entry_idx in tqdm(
        range(1, len(original_spectra)),
        desc="Creating mixtures"
    ):

        wn1, abs1 = original_spectra[entry_idx]
        entry_name = entry_names[entry_idx]

        for mix_idx in range(nMixes):

            source_idx = rng.integers(0, entry_idx)

            wn2, abs2 = original_spectra[source_idx]

            percentage = rng.integers(1, 100) / 100

            mixed_wn, mixed_abs = mix_spectra(wn1, abs1, wn2, abs2, percentage, rng)

            mixed_headers.extend([
                "",
                (
                    f"{entry_name}_{percentage:.0%} + "
                    f"{entry_names[source_idx]}_"
                    f"{1 - percentage:.0%}"
                )
            ])

            mixed_subheaders.extend([
                "Wavenumber [1/cm]",
                "Absorbance"
            ])

            mixed_columns.extend([
                mixed_wn,
                mixed_abs
            ])


    pure_array = pad_columns(pure_columns)

    pure_path = file.replace(
        ".csv",
        "_scrambled_pure.csv"
    ).replace(
        "/data/",
        "/data/scrambled/pure/"
    )

    print("Saving pure data...")

    with open(pure_path, "w", newline="") as f:

        f.write(",".join(pure_headers) + "\n")
        f.write(",".join(pure_subheaders) + "\n")

        np.savetxt(
            f,
            pure_array,
            delimiter=","
        )

    mixed_array = pad_columns(mixed_columns)

    mixed_path = file.replace(
        ".csv",
        "_scrambled_mixed.csv"
    ).replace(
        "/data/",
        "/data/scrambled/mixed/"
    )

    print("Saving mixed data...")

    with open(mixed_path, "w", newline="") as f:

        f.write(",".join(mixed_headers) + "\n")
        f.write(",".join(mixed_subheaders) + "\n")

        np.savetxt(
            f,
            mixed_array,
            delimiter=","
        )
