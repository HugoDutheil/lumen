from dataloader import*
from generator import*
from model import*
from model_gated import*
from model_mlp import*
from model_autoencoder import*
from PCA_aided_pipe import*
from ast import main
import argparse
import yaml

# For the path, we go to the root of the current project and then to the data folder, where we have the data.csv file.  
file_path = '../data/PSLC.csv'
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def main():
    parser = argparse.ArgumentParser(description="Run training process")
    parser.add_argument(
        "--config",
        type=str,
        default="../conf.yaml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    df_clean = parse_csv(file_path)
    #print(df_clean.shape)  # Output: (N_samples, N_wavenumbers)
    #print(df_clean.index[0:2])

    # ------------------------------- Get data --------------------------------------------------------------
    labels = df_clean.index.to_numpy()
    # Prepare the data, mix components and add noise
    _, mixed_samples, mixed_targets = data_generator(df_clean, num_samples=config['training'].get('num_samples'))

    # ------------------------------- Split data in training and test datasets -------------------------------
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
    #train_split = config['training'].get('train_split')
    #split_idx = int(train_split * len(mixed_samples))
    # train_data = np.array(mixed_samples[:split_idx])
    # train_labels = np.array(mixed_targets[:split_idx])
    # test_data = np.array(mixed_samples[split_idx:])
    # test_labels = np.array(mixed_targets[split_idx:])
    train_data, train_labels = X[train_idx], y[train_idx]
    test_data,  test_labels  = X[test_idx],  y[test_idx]
    val_data, val_labels = X[val_idx], y[val_idx]

    weights_pth = config['experiment'].get('weights_filename')
    weights_class = config['experiment'].get('weights_classifier')
    print(f"Weights will be saved at {weights_pth}")
    num_epochs = config['training'].get('n_epochs')

    # ------------------------------- Prepare training --------------------------------------------------------------
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = config['model'].get('type')
    save_results = config['experiment']['save_results']

    hidden_dim = config['model'].get('hidden_dim')
    lr = config['training'].get('lr')
    model = None
    criterion = None
    detection_criterion = None
    concentration_criterion = None
    match model_name:
        case "cnn":
            model = CNN(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
            model_single_head = CNNSingleHead(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
            #model.to(device)

            #Single-head model
            criterion = nn.KLDivLoss(reduction="batchmean")

            #Two-heads model 
            detection_criterion = nn.BCELoss()
            concentration_criterion = nn.MSELoss() #nn.KLDivLoss(reduction="batchmean")  # Using KL Divergence Loss for multi-label classification
            #Notice that KLDivLoss averages over every element by default, not per sample, that is why we use batchmean instead.
            optimizer = torch.optim.Adam(model_single_head.parameters(), lr=0.001)
            optimizer_gated = torch.optim.Adam(model.parameters(), lr=float(lr))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_gated, 
            mode='min',         # 'min' because we want to minimize the loss
            factor=0.5,         # Multiply learning rate by 0.5 when triggered
            patience=5       # Prints a message when learning rate changes
        )
            train(train_data, train_labels, model, detection_criterion, concentration_criterion, optimizer_gated, num_epochs=num_epochs)
            if save_results:
                torch.save(model.state_dict(), weights_pth)
            test_global(test_data, test_labels, labels, model, weights_pth="../weights/double_head_corrected_loss.pth", hidden_dim=hidden_dim)
    #MLP model
        case "mlp":
            model_MLP = MLP(input_dim=len(train_data[0]))
            criterion_mlp = nn.MSELoss()
            detection_criterion = nn.BCELoss()
            concentration_criterion = nn.MSELoss()
            optimizer_mlp = torch.optim.Adam(model_MLP.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_mlp, 
                mode='min',
                factor=0.5,
                patience=5
            )
            ##Training with MLP model  
            train_mlp(train_data, train_labels, model_MLP, detection_criterion, concentration_criterion, optimizer_mlp, num_epochs=num_epochs)
            if save_results:
                torch.save(model_MLP.state_dict(), weights_pth)
            test_mlp(test_data, test_labels, model_MLP, weights_pth=weights_pth)

    #AutoEncoder model
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

    #PCA Pipeline
    #parser = Parser
        case "pca_mlp":
            pca_dim = config['model'].get('pca_dim')
            model_PCA = MLP(input_dim=pca_dim, hidden_dim=hidden_dim, compounds=144, channels=1).to(device)
            pca_pipe = PCAPipeline(model_PCA, pca_dim=pca_dim)
            train_pca, test_pca = run_pca(pca_pipe, train_data, test_data)
            #analyze_pca(train_pca, train_labels)
            pos_weight = torch.tensor([7.0])
            detection_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            concentration_criterion = nn.L1Loss()
            criterion_mlp = nn.L1Loss() #nn.MSELoss()
            optimizer_mlp = torch.optim.Adam(model_PCA.parameters(), lr = lr)
            train_mlp(train_pca, train_labels, model_PCA, detection_criterion, concentration_criterion, optimizer_mlp, num_epochs=num_epochs)
            if save_results:
                torch.save(model_PCA.state_dict(), weights_pth)
            test_mlp(test_pca, test_labels, model_PCA, weights_pth=weights_pth, hidden_dim=hidden_dim, presence_threshold=0.5)

        case _:
            raise ValueError(f"Unknown model type {model} specified in conf.yaml")
   
if __name__ == "__main__":
    main()