from dataloader import*
from generator import*
from model import*
from model_gated import*
from model_mlp import*
from model_autoencoder import*
from PCA_aided_pipe import*
from nnls import*                                                   
from onthefly import OnTheFlyMixtures, generate_batch, ScalerPCA, train_onthefly   
from ast import main
import argparse
import yaml
import os
import joblib                                                            
from sklearn.preprocessing import StandardScaler                          
from sklearn.decomposition import PCA                                    
from torch.utils.data import DataLoader                                  
from pathlib import Path

# For the path, we go to the root of the current project and then to the data folder, where we have the data.csv file.
ROOT = Path(__file__).resolve().parents[3]
CONF_PATH = ROOT / "conf.yaml"
DATA_PATH = ROOT / "data"
WEIGHTS_PATH = ROOT / "weights"
file_path = DATA_PATH #/ "PSLC.csv"
config_path = CONF_PATH
print(file_path, config_path)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def fixed_eval_sets(pure, wn, config):
    n_val = config['training'].get('val_samples', 5000)
    n_test = config['training'].get('test_samples', 20000)
    X_val, y_val = generate_batch(pure, wn, n_val, np.random.default_rng(1))
    X_test, y_test = generate_batch(pure, wn, n_test, np.random.default_rng(2))
    return X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Run training process")
    parser.add_argument(
        "--config",
        type=str,
        default=CONF_PATH,
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    data_path = DATA_PATH / config['experiment'].get('data_path')
    data_cfg = config.get('data', {})
    df_clean = parse_csv(data_path, wn_range=data_cfg.get('wn_range'), step=data_cfg.get('wn_step', 1.0))

    labels = df_clean.index.to_numpy()
    pure = df_clean.values.astype(np.float32)                 # NEW: (n_compounds, n_points) reference spectra
    wn = df_clean.columns.astype(float).to_numpy()            # NEW: wavenumber axis
    n_compounds = len(pure)

    weights_pth = WEIGHTS_PATH / config['experiment'].get('weights_filename')
    weights_class = WEIGHTS_PATH / config['experiment'].get('weights_classifier')
    print(f"Weights will be saved at {weights_pth}")
    num_epochs = config['training'].get('n_epochs')
    model_name = config['model'].get('type')                  # moved up: decides which data to build
    save_results = config['experiment']['save_results']
    hidden_dim = config['model'].get('hidden_dim')
    lr = config['training'].get('lr')

    # ------------------------------- Get data --------------------------------------------------------------
    if model_name in ("pca_mlp", "nnls"):
        X_val, y_val, X_test, y_test = fixed_eval_sets(pure, wn, config)
    else:
        # Unchanged path for cnn / mlp / autoencoder
        _, mixed_samples, mixed_targets = data_generator(df_clean, num_samples=config['training'].get('num_samples'))
        rng = np.random.default_rng(42)
        X = np.array(mixed_samples)
        y = np.array(mixed_targets)
        perm = rng.permutation(len(X))
        train_prop = config['training']['train_split']
        val_prop = config['training']['val_split']
        test_prop = 1 - train_prop - val_prop
        split_idx_train = int(train_prop * len(X))
        split_idx_test = split_idx_train + int(test_prop*len(X))
        split_idx_val = split_idx_test + int(val_prop * len(X))
        train_idx, test_idx, val_idx = perm[:split_idx_train], perm[split_idx_train:split_idx_val], perm[split_idx_val:]
        train_data, train_labels = X[train_idx], y[train_idx]
        test_data,  test_labels  = X[test_idx],  y[test_idx]
        val_data, val_labels = X[val_idx], y[val_idx]

    # ------------------------------- Prepare training --------------------------------------------------------------
    model = None
    criterion = None
    detection_criterion = None
    concentration_criterion = None
    match model_name:
        case "cnn":
            model = CNN(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
            model_single_head = CNNSingleHead(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
            criterion = nn.KLDivLoss(reduction="batchmean")
            detection_criterion = nn.BCELoss()
            concentration_criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model_single_head.parameters(), lr=0.001)
            optimizer_gated = torch.optim.Adam(model.parameters(), lr=float(lr))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_gated, mode='min', factor=0.5, patience=5)
            train(train_data, train_labels, model, detection_criterion, concentration_criterion, optimizer_gated, num_epochs=num_epochs)
            if save_results:
                torch.save(model.state_dict(), weights_pth)
            test_global(test_data, test_labels, labels, model, weights_pth="../weights/double_head_corrected_loss.pth", hidden_dim=hidden_dim)

        case "mlp":
            model_MLP = MLP(input_dim=len(train_data[0]))
            criterion_mlp = nn.MSELoss()
            detection_criterion = nn.BCELoss()
            concentration_criterion = nn.MSELoss()
            optimizer_mlp = torch.optim.Adam(model_MLP.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mlp, mode='min', factor=0.5, patience=5)
            train_mlp(train_data, train_labels, model_MLP, detection_criterion, concentration_criterion, optimizer_mlp, num_epochs=num_epochs)
            if save_results:
                torch.save(model_MLP.state_dict(), weights_pth)
            test_mlp(test_data, test_labels, model_MLP, weights_pth=weights_pth)

        case "autoencoder":
            input_dim=len(train_data[0])
            encoder = Encoder(input_dim=len(train_data[0]), latent_dim=144, hidden_dim=512)
            decoder = Decoder(input_dim=input_dim, latent_dim=144, hidden_dim=512)
            criterion_auto = nn.MSELoss()
            model_auto = Autoencoder(encoder, decoder, input_dim=input_dim, latent_dim=144, hidden_dim=512)
            classifier = Classifier(encoder, input_dim=input_dim, latent_dim=144, hidden_dim=512)
            train_autoencoder(train_data, model_auto, criterion_auto, num_epochs=num_epochs)
            if save_results:
                torch.save(encoder.state_dict(), weights_pth)
            train_classifier(train_data, train_labels, classifier, weights_pth, detection_criterion, concentration_criterion, num_epochs=num_epochs)
            if save_results:
                torch.save(classifier.state_dict(), weights_class)
            test_classifier(test_data, test_labels, classifier, weights_pth=weights_class)

        case "pca_scratch_mlp":
            train_model = config['training'].get('train', True)
            run_weights = weights_pth if save_results else os.path.splitext(weights_pth)[0] + "_lastrun.pth"
            mtx_path = ROOT / config['model'].get('mtx_path', 'data/pca_basis.mtx')
            basis_from = config['model'].get('basis_from', 'dataloader')     # dataloader | csv

            # 1) Basis from the PCA pipeline (.mtx), on the same wavenumber range as the DL data
            Vt, basis_grid = pca_baseline_basis(data_path, mtx_path, wn,
                                                df_clean=df_clean if basis_from == 'dataloader' else None,
                                                retrain=train_model)
            pca_dim = min(config['model'].get('pca_dim'), Vt.shape[0])   # at most one component per reference
            print(f"Using {pca_dim} components of the PCA pipeline's basis")

            # 2) Mixtures -> scores on that basis, standardized (the uncentered basis gives scores of very
            #    different sizes: the first one is ~200x the 100th). Mean/std are saved with the weights.
            train_pca = project_on_basis(train_data, wn, Vt, basis_grid, pca_dim)
            test_pca = project_on_basis(test_data, wn, Vt, basis_grid, pca_dim)
            scaler_path = os.path.splitext(run_weights if train_model else weights_pth)[0] + "_scores_scaler.npz"
            if train_model:
                np.savez(scaler_path, mean=train_pca.mean(0), std=train_pca.std(0) + 1e-8)
            sc = np.load(scaler_path)
            train_pca = ((train_pca - sc["mean"]) / sc["std"]).astype(np.float32)
            test_pca = ((test_pca - sc["mean"]) / sc["std"]).astype(np.float32)

            # 3) MLP, trained and tested with the existing functions
            model_PCA = MLP(input_dim=pca_dim, hidden_dim=hidden_dim, compounds=n_compounds, channels=1).to(device)
            detection_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(config['training'].get('pos_weight', 1.0))]))
            concentration_criterion = nn.L1Loss()
            if train_model:
                optimizer_mlp = torch.optim.Adam(model_PCA.parameters(), lr=float(lr))
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_mlp, mode='min', factor=0.5, patience=5)
                train_mlp(train_pca, train_labels, model_PCA, detection_criterion, concentration_criterion,
                          optimizer_mlp, num_epochs=num_epochs, scheduler=scheduler)
                torch.save(model_PCA.state_dict(), run_weights)
            test_mlp(test_pca, test_labels, model_PCA, weights_pth=run_weights if train_model else weights_pth,
                     hidden_dim=hidden_dim, presence_threshold=config['model'].get('presence_threshold', 0.5),
                     labels=labels)
          

        case "pca_mlp":
            pca_dim = config['model'].get('pca_dim')
            transform_path = os.path.splitext(weights_pth)[0] + "_transform.joblib"
            train_model = config['training'].get('train', True)
            run_weights = weights_pth if save_results else os.path.splitext(weights_pth)[0] + "_lastrun.pth"

            # Scaler + PCA: fitted once on a fixed batch (seed 0) when training, reloaded otherwise
            if train_model:
                X_fit, _ = generate_batch(pure, wn, config['training'].get('fit_samples', 20000), np.random.default_rng(0))
                scaler = StandardScaler().fit(X_fit)
                pca = PCA(n_components=pca_dim).fit(scaler.transform(X_fit))
                transform = ScalerPCA(scaler, pca)
                joblib.dump(transform, os.path.splitext(run_weights)[0] + "_transform.joblib")
                del X_fit
            else:
                transform = joblib.load(transform_path)
            Z_val = transform(X_val).astype(np.float32)
            Z_test = transform(X_test).astype(np.float32)

            # Model and loss
            model_PCA = MLP(input_dim=pca_dim, hidden_dim=hidden_dim, compounds=n_compounds, channels=1).to(device)
            pos_weight = torch.tensor([float(config['training'].get('pos_weight', 1.0))])
            detection_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            concentration_criterion = nn.L1Loss()
            # model outputs (probs, gated_concs, probs_raw, concs_raw): loss uses the logits and raw concs
            loss_fn = lambda out, y: compute_loss(out[2], out[3], y, detection_criterion, concentration_criterion)

            # Training on fresh batches every step
            if train_model:
                optimizer_mlp = torch.optim.Adam(model_PCA.parameters(), lr=float(lr))
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_mlp, mode='min', factor=0.5, patience=5)
                train_stream = OnTheFlyMixtures(pure, wn, batch_size=config['training'].get('batch_size', 256),
                                                seed=3, transform=transform)
                train_loader = DataLoader(train_stream, batch_size=None,
                                          num_workers=config['training'].get('num_workers', 2))
                train_onthefly(model_PCA, train_loader, Z_val, y_val, loss_fn, optimizer_mlp, scheduler,
                               num_epochs=num_epochs, steps_per_epoch=config['training'].get('steps_per_epoch', 200),
                               device=device, best_path=run_weights)

            # Presence threshold chosen on validation set, then a single evaluation on the test set
            eval_weights = run_weights if train_model else weights_pth
            model_PCA.load_state_dict(torch.load(eval_weights, map_location=device))
            model_PCA.eval()
            with torch.no_grad():
                p_val = model_PCA(torch.tensor(Z_val, device=device))[0].cpu().numpy()
            threshold = tune_threshold(p_val, y_val, grid=np.linspace(0.05, 0.95, 91))
            print(f"Presence threshold chosen on validation: {threshold:.2f}")
            test_mlp(Z_test, y_test, model_PCA, weights_pth=eval_weights, hidden_dim=hidden_dim,
                     presence_threshold=threshold, labels=labels)

    # NNLS physics baseline: no training, only a z-score threshold chosen on validation
        case "nnls":
            nn_cfg = config.get('nnls', {})
            sum_to_one = nn_cfg.get('sum_to_one', True)
            n_jobs = nn_cfg.get('n_jobs', -1)
            n_val, n_test = nn_cfg.get('val_samples', 2000), nn_cfg.get('test_samples', 5000)
            print(f"Fitting NNLS on {n_val} validation spectra...")
            C_val, Zs_val = nnls_batch(X_val[:n_val], pure, wn, sum_to_one=sum_to_one, n_jobs=n_jobs)
            z_threshold = tune_threshold(Zs_val, y_val[:n_val])
            print(f"z-score threshold chosen on validation: {z_threshold:.2f}")
            print(f"Fitting NNLS on {n_test} test spectra...")
            C_test, Zs_test = nnls_batch(X_test[:n_test], pure, wn, sum_to_one=sum_to_one, n_jobs=n_jobs)
            detected, conc = predict_nnls(C_test, Zs_test, z_threshold)
            report(detected, conc, y_test[:n_test], title="NNLS (z-score) REPORT")

        case _:
            raise ValueError(f"Unknown model type {model_name} specified in conf.yaml")


if __name__ == "__main__":
    main()