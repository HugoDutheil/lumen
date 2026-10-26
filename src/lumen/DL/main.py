# from dataloader import*
# from generator import*
# from model import*
# from model_gated import*
# from model_mlp import*
# from model_autoencoder import*
# from PCA_aided_pipe import*
# from ast import main
# import argparse
# import yaml
# import os

# # PROBLEMS : 
# # - presence_threshold does not seem to be detected
# # - model does not scale anymore (1999 : 72.8% of top-1 match, drops to 18.4% with 3999)
# # - thousands of hallucinations and missed components


# # For the path, we go to the root of the current project and then to the data folder, where we have the data.csv file.  
# file_path = '../data/PSLC.csv'
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = torch.device("cpu")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

# print(f"Using device: {device}")

# def main():
#     parser = argparse.ArgumentParser(description="Run training process")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="../conf.yaml",
#         help="Path to the configuration file"
#     )
#     args = parser.parse_args()
#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)
#     df_clean = parse_csv(file_path)
#     #print(df_clean.shape)  # Output: (N_samples, N_wavenumbers)
#     #print(df_clean.index[0:2])

#     # ------------------------------- Get data --------------------------------------------------------------
#     labels = df_clean.index.to_numpy()
#     # Prepare the data, mix components and add noise
#     _, mixed_samples, mixed_targets = data_generator(df_clean, num_samples=config['training'].get('num_samples'))

#     # ------------------------------- Split data in training and test datasets -------------------------------
#     rng = np.random.default_rng(42)
#     X = np.array(mixed_samples)
#     y = np.array(mixed_targets)
#     perm = rng.permutation(len(X))
#     train_prop = config['training']['train_split']
#     val_prop = config['training']['val_split']
#     test_prop = 1 - train_prop - val_prop


#     split_idx_train = int(train_prop * len(X))
#     split_idx_test = split_idx_train + int(test_prop*len(X))
#     split_idx_val = split_idx_test + int(val_prop * len(X))

#     train_idx, test_idx, val_idx = perm[:split_idx_train], perm[split_idx_train:split_idx_val], perm[split_idx_val:]
#     #train_split = config['training'].get('train_split')
#     #split_idx = int(train_split * len(mixed_samples))
#     # train_data = np.array(mixed_samples[:split_idx])
#     # train_labels = np.array(mixed_targets[:split_idx])
#     # test_data = np.array(mixed_samples[split_idx:])
#     # test_labels = np.array(mixed_targets[split_idx:])
#     train_data, train_labels = X[train_idx], y[train_idx]
#     test_data,  test_labels  = X[test_idx],  y[test_idx]
#     val_data, val_labels = X[val_idx], y[val_idx]

#     weights_pth = config['experiment'].get('weights_filename')
#     weights_class = config['experiment'].get('weights_classifier')
#     print(f"Weights will be saved at {weights_pth}")
#     num_epochs = config['training'].get('n_epochs')

#     # ------------------------------- Prepare training --------------------------------------------------------------
#     #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     model_name = config['model'].get('type')
#     save_results = config['experiment']['save_results']

#     hidden_dim = config['model'].get('hidden_dim')
#     lr = config['training'].get('lr')
#     model = None
#     criterion = None
#     detection_criterion = None
#     concentration_criterion = None
#     match model_name:
#         case "cnn":
#             model = CNN(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
#             model_single_head = CNNSingleHead(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
#             #model.to(device)

#             #Single-head model
#             criterion = nn.KLDivLoss(reduction="batchmean")

#             #Two-heads model 
#             detection_criterion = nn.BCELoss()
#             concentration_criterion = nn.MSELoss() #nn.KLDivLoss(reduction="batchmean")  # Using KL Divergence Loss for multi-label classification
#             #Notice that KLDivLoss averages over every element by default, not per sample, that is why we use batchmean instead.
#             optimizer = torch.optim.Adam(model_single_head.parameters(), lr=0.001)
#             optimizer_gated = torch.optim.Adam(model.parameters(), lr=float(lr))
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer_gated, 
#             mode='min',         # 'min' because we want to minimize the loss
#             factor=0.5,         # Multiply learning rate by 0.5 when triggered
#             patience=5       # Prints a message when learning rate changes
#         )
#             train(train_data, train_labels, model, detection_criterion, concentration_criterion, optimizer_gated, num_epochs=num_epochs)
#             if save_results:
#                 torch.save(model.state_dict(), weights_pth)
#             test_global(test_data, test_labels, labels, model, weights_pth="../weights/double_head_corrected_loss.pth", hidden_dim=hidden_dim)
#     #MLP model
#         case "mlp":
#             model_MLP = MLP(input_dim=len(train_data[0]))
#             criterion_mlp = nn.MSELoss()
#             detection_criterion = nn.BCELoss()
#             concentration_criterion = nn.MSELoss()
#             optimizer_mlp = torch.optim.Adam(model_MLP.parameters(), lr = lr)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer_mlp, 
#                 mode='min',
#                 factor=0.5,
#                 patience=5
#             )
#             ##Training with MLP model  
#             train_mlp(train_data, train_labels, model_MLP, detection_criterion, concentration_criterion, optimizer_mlp, num_epochs=num_epochs)
#             if save_results:
#                 torch.save(model_MLP.state_dict(), weights_pth)
#             test_mlp(test_data, test_labels, model_MLP, weights_pth=weights_pth)

#     #AutoEncoder model
#         case "autoencoder":
#             input_dim=len(train_data[0])
#             encoder = Encoder(input_dim=len(train_data[0]), latent_dim=144, hidden_dim=512)
#             decoder = Decoder(input_dim=input_dim, latent_dim=144, hidden_dim=512)
#             criterion_auto = nn.MSELoss()
#             model_auto = Autoencoder(encoder, decoder, input_dim=input_dim, latent_dim=144, hidden_dim=512)
#             classifier = Classifier(encoder, input_dim=input_dim, latent_dim=144, hidden_dim=512)

#             train_autoencoder(train_data, model_auto, criterion_auto, num_epochs=num_epochs)
#             if save_results:
#                 torch.save(encoder.state_dict(), weights_pth)
#             train_classifier(train_data, train_labels, classifier, weights_pth, detection_criterion, concentration_criterion, num_epochs=num_epochs)
#             if save_results:
#                 torch.save(classifier.state_dict(), weights_class)
#             test_classifier(test_data, test_labels, classifier, weights_pth=weights_class)

#     #PCA Pipeline
#     #parser = Parser
#         case "pca_mlp":
#             pca_dim = config['model'].get('pca_dim')
#             pca_path = os.path.splitext(weights_pth)[0] + "_pca.joblib"
#             train_model = config['training'].get('train', True)
#             model_PCA = MLP(input_dim=pca_dim, hidden_dim=hidden_dim, compounds=144, channels=1).to(device)
#             pca_pipe = PCAPipeline(model_PCA, pca_dim=pca_dim)
#             train_pca, test_pca = run_pca(
#                 pca_pipe, train_data, test_data,
#                 pca_path=pca_path if (train_model and save_results) or not train_model else None,
#                 fit=train_model,
#             )
#             #analyze_pca(train_pca, train_labels)
#             pos_weight = torch.tensor([7.0])
#             detection_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#             concentration_criterion = nn.L1Loss()
#             criterion_mlp = nn.L1Loss() #nn.MSELoss()
#             optimizer_mlp = torch.optim.Adam(model_PCA.parameters(), lr = lr)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_mlp, mode='min', factor=0.5, patience=5)
#             if train_model:
#                 train_mlp(train_pca, train_labels, model_PCA, detection_criterion, concentration_criterion,
#                           optimizer_mlp, num_epochs=num_epochs, scheduler=scheduler)
#                 if save_results:
#                     torch.save(model_PCA.state_dict(), weights_pth)
#             test_mlp(test_pca, test_labels, model_PCA, weights_pth=weights_pth, hidden_dim=hidden_dim,
#                      presence_threshold=0.4, labels=labels)

#         case _:
#             raise ValueError(f"Unknown model type {model} specified in conf.yaml")
   
# if __name__ == "__main__":
#     main()
from dataloader import*
from generator import*
from model import*
from model_gated import*
from model_mlp import*
from model_autoencoder import*
from PCA_aided_pipe import*
from nnls import*                                                    # NEW
from onthefly import OnTheFlyMixtures, generate_batch, ScalerPCA, train_onthefly   # NEW
from ast import main
import argparse
import yaml
import os
import joblib                                                             # NEW
from sklearn.preprocessing import StandardScaler                          # NEW
from sklearn.decomposition import PCA                                     # NEW
from torch.utils.data import DataLoader                                   # NEW
from pathlib import Path

# For the path, we go to the root of the current project and then to the data folder, where we have the data.csv file.
ROOT = Path(__file__).resolve().parents[3]
CONF_PATH = ROOT / "conf.yaml"
DATA_PATH = ROOT / "data"
WEIGHTS_PATH = ROOT / "weights"
file_path = DATA_PATH / "PSLC.csv"
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
    """NEW: validation and test sets, each from its own seed, never used for training."""
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
    df_clean = parse_csv(file_path)

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