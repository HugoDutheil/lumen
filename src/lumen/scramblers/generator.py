import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def noise(signal, snr_db, seed=42):
    """Add Gaussian noise to signal so the result has the given SNR (dB)"""
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal+noise+ shift(signal, seed)

def shift(signal, seed=42):
    norm = 2 * (signal - 400) / (4000-400) - 1
    amplitude = 0.01 * 0.05
    a, b, c = np.random.uniform(-amplitude, amplitude), np.random.uniform(-amplitude, amplitude), np.random.uniform(-amplitude, amplitude)
    shift = a + b * norm + c * norm**2
    return shift

def baseline(wavenumbers, max_amplitude, rng):
    """Random smooth background: offset + tilt + curvature across the spectral range."""
    wn = np.asarray(wavenumbers, dtype=np.float32)
    x = 2 * (wn - wn.min()) / (wn.max() - wn.min()) - 1       # wavenumber rescaled to [-1, 1]
    a, b, c = rng.uniform(-max_amplitude, max_amplitude, size=3)
    return (a + b * x + c * x**2).astype(np.float32)

def add_noise(signal, snr_db, rng):
    """Gaussian noise such that the result has the given signal-to-noise ratio (dB)."""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return signal + rng.normal(0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)



def data_generator(data, num_samples=200, seed=42, snr_range_db=(10,40), n_components=(2,8), baseline_frac=0.05):
    """
    A generator function that generates mixes of data from the Panda DataFrame, adding some Gaussian noise to each sample
    """
    rng = np.random.default_rng(seed)                    # local RNG: reproducible, independent of global state
    pure = data.values.astype(np.float32)                # (n_compounds, n_points)
    wavenumbers = data.columns.astype(float).to_numpy()  # assumes the columns are the wavenumbers
    n_compounds = len(pure)
    np.random.seed(seed)
    data_noise = data.copy()
    data_noise = data.values.astype(np.float32)
    labels = data.index.to_numpy()
    # Adding Gaussian noise to the samples.
    # The following code is too naive, it adds noise regardless of the signal's amplitude
    #data_noise = data_noise + np.random.normal(0, 0.1, size=data_noise.shape)

    # Generating mixtures of the sample, along with the labels of the original samples
    mixed_samples_wo_noise = []
    mixed_samples = []
    mixed_targets = []
    mix = None
    for _ in range(num_samples):
        # # Randomly select between 2 and 5 samples to mix
        # num_to_mix = np.random.randint(2, 5)
        # indices = np.random.choice(len(labels), num_to_mix, replace=False)
        # weights = np.zeros(len(labels), dtype=np.float32)
        # alpha = np.ones(num_to_mix) * 0.5
        # raw_weights = np.random.dirichlet(alpha).astype(np.float32)
        # for idx, w in zip(indices, raw_weights):
        #     weights[idx] = w
        # mix = np.average(data_noise, axis=0, weights=weights)
        # mixed_samples_wo_noise.append(mix)
        # snr_db = np.random.uniform(*snr_range_db)
        # mix = noise(mix, snr_db).astype(np.float32)
        # #mix += np.random.normal(0, 0.05, size=data_noise[1].shape)  # Add some noise to the mixed sample
        # mixed_samples.append(mix)
        # mixed_targets.append(weights)
        k = rng.integers(n_components[0], n_components[1] + 1)
        idx = rng.choice(n_compounds, size=k, replace=False)
        weights = np.zeros(n_compounds, dtype=np.float32)
        weights[idx] = rng.dirichlet(np.full(k, 1.0)).astype(np.float32)
        mix = weights @ pure
        mix = mix + baseline(wavenumbers, baseline_frac * np.abs(mix).max(), rng)
        mix = add_noise(mix, rng.uniform(*snr_range_db), rng)

        mixed_samples.append(mix)
        mixed_targets.append(weights)

    #plt.figure(figsize=(10, 5))
    #print(len(mix))
    #plt.plot(np.arange(len(mix)), noise(mixed_samples[0], snr_db).astype(np.float32), label='Noisy spectre')
    #plt.plot(np.arange(len(mix)), mixed_samples_wo_noise[0], marker='o', label='True mix')
    #plt.title('Training Loss over Epochs')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.grid()
    #plt.show()

    return data_noise, mixed_samples, mixed_targets
    