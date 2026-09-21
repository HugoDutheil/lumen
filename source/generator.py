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


def data_generator(data, num_samples=200, seed=42, snr_range_db=(10,40)):
    """
    A generator function that generates mixes of data from the Panda DataFrame, adding some Gaussian noise to each sample
    """
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
        # Randomly select between 2 and 5 samples to mix
        num_to_mix = np.random.randint(2, 5)
        indices = np.random.choice(len(labels), num_to_mix, replace=False)
        weights = np.zeros(len(labels), dtype=np.float32)
        alpha = np.ones(num_to_mix) * 0.5
        raw_weights = np.random.dirichlet(alpha).astype(np.float32)
        for idx, w in zip(indices, raw_weights):
            weights[idx] = w
        mix = np.average(data_noise, axis=0, weights=weights)
        mixed_samples_wo_noise.append(mix)
        snr_db = np.random.uniform(*snr_range_db)
        mix = noise(mix, snr_db).astype(np.float32)
        #mix += np.random.normal(0, 0.05, size=data_noise[1].shape)  # Add some noise to the mixed sample
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
    