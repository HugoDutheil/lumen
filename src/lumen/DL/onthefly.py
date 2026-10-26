import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import tqdm 
 
 
def generate_batch(pure, wavenumbers, batch_size, rng,
                   n_components=(2, 8), snr_range_db=(10, 40), baseline_frac=0.05, alpha=1.0):
    """Vectorized version of data_generator: same recipe, a whole batch at once."""
    n_compounds, n_points = pure.shape
    # how many compounds in each mixture, and which ones
    k = rng.integers(n_components[0], n_components[1] + 1, size=batch_size)
    ranks = np.argsort(rng.random((batch_size, n_compounds)), axis=1)    # random order of compounds per row
    chosen = ranks < k[:, None]                                          # first k of that order -> present
    # Dirichlet(alpha) weights on the chosen compounds = normalized Gamma(alpha) draws
    g = rng.gamma(alpha, size=(batch_size, n_compounds)) * chosen
    weights = (g / g.sum(axis=1, keepdims=True)).astype(np.float32)
    mix = weights @ pure                                                 # (batch, n_points)
    # smooth baseline: offset + tilt + curvature, scaled to each spectrum's max
    x = 2 * (wavenumbers - wavenumbers.min()) / (wavenumbers.max() - wavenumbers.min()) - 1
    amp = baseline_frac * np.abs(mix).max(axis=1, keepdims=True)
    a, b, c = (rng.uniform(-1, 1, size=(3, batch_size, 1)) * amp)
    mix = mix + a + b * x + c * x**2
    # Gaussian noise at a random SNR per spectrum
    snr_db = rng.uniform(*snr_range_db, size=(batch_size, 1))
    noise_std = np.sqrt(np.mean(mix**2, axis=1, keepdims=True) / 10 ** (snr_db / 10))
    mix = mix + rng.normal(size=mix.shape) * noise_std
    return mix.astype(np.float32), weights
 
 
class OnTheFlyMixtures(IterableDataset):
    """Endless stream of fresh mixtures. `transform` maps a raw batch to model inputs (e.g. scaler+PCA)."""
    def __init__(self, pure, wavenumbers, batch_size=256, seed=0, transform=None, **gen_kwargs):
        self.pure = pure.astype(np.float32)
        self.wn = np.asarray(wavenumbers, dtype=np.float64)
        self.batch_size, self.seed, self.transform, self.gen_kwargs = batch_size, seed, transform, gen_kwargs
 
    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, worker_id])   # each worker gets its own stream
        while True:
            X, Y = generate_batch(self.pure, self.wn, self.batch_size, rng, **self.gen_kwargs)
            if self.transform is not None:
                X = self.transform(X)
            yield torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(Y)
 
 
class ScalerPCA:
    """Picklable transform (needed for num_workers > 0 on macOS): standardize, then project with PCA."""
    def __init__(self, scaler, pca):
        self.scaler, self.pca = scaler, pca
 
    def __call__(self, X):
        return self.pca.transform(self.scaler.transform(X))
 
 
def train_onthefly(model, train_loader, Z_val, y_val, loss_fn, optimizer, scheduler=None,
                   num_epochs=100, steps_per_epoch=200, device="cpu", best_path=None):
    """
    Train on an endless stream of fresh batches.
    loss_fn(model_outputs, y) -> scalar loss, e.g.
        lambda out, y: compute_loss(out[2], out[3], y, detection_criterion, concentration_criterion)
    The scheduler steps on the VALIDATION loss; the best model (lowest val loss) is saved to best_path.
    """
    Zv = torch.as_tensor(Z_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    batches = iter(train_loader)
    best = float("inf")
    history = []
    #for epoch in range(num_epochs):
    for epoch in enumerate(tqdm(num_epochs)):
        model.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = next(batches)
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
 
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Zv), yv).item()
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append((running / steps_per_epoch, val_loss))
        print(f"Epoch {epoch+1}/{num_epochs}  train {running/steps_per_epoch:.4f}  val {val_loss:.4f}")
        if best_path is not None and val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), best_path)
    return history