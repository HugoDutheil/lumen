from dataloader import*
from generator import*
from model import*
from ast import main
import argparse
import yaml

# For the path, we go to the root of the current project and then to the data folder, where we have the data.csv file.  
file_path = '../data/PSLC.csv'

def main():
    parser = argparse.ArgumentParser(description="Run training process")
    parser.add_argument(
        "--config",
        type=str,
        default="conf.yaml",
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
    #TODO : Sample uniformly the data used for training
    train_split = config['training'].get('train_split')
    split_idx = int(train_split * len(mixed_samples))
    train_data = np.array(mixed_samples[:split_idx])
    train_labels = np.array(mixed_targets[:split_idx])
    test_data = np.array(mixed_samples[split_idx:])
    test_labels = np.array(mixed_targets[split_idx:])

    # ------------------------------- Prepare training --------------------------------------------------------------
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    hidden_dim = config['model'].get('hidden_dim')
    model = CNN(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
    model_single_head = CNNSingleHead(input_channels=1, hidden_dim=hidden_dim, compounds=len(mixed_targets[0]))
    #model.to(device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    detection_criterion = nn.BCEWithLogitsLoss()
    concentration_criterion = nn.MSELoss() #nn.KLDivLoss(reduction="batchmean")  # Using KL Divergence Loss for multi-label classification
    #Notice that KLDivLoss averages over every element by default, not per sample, that is why we use batchmean instead.
    optimizer = torch.optim.Adam(model_single_head.parameters(), lr=0.001)
    
    # ------------------------------- Training and testing --------------------------------------------------------------
    #Training with 2-heads model 
    weights_pth = config['experiment'].get('weights_filename')
    #train(train_data, train_labels, model, detection_criterion, concentration_criterion, optimizer, num_epochs=10)
    #test_global(test_data, test_labels, labels, model)
    #torch.save(model.state_dict(), weights_pth)

    #Training with one-head model 
    #train_single_head(train_data, train_labels, model_single_head, criterion, optimizer, num_epochs=100)
    #torch.save(model_single_head.state_dict(), weights_pth)
    test_single_head(test_data, test_labels, len(mixed_targets[0]), model_single_head)

if __name__ == "__main__":
    main()