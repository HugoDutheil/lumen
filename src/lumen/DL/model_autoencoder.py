import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model_mlp import*

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, input_dim=3601, latent_dim=144, hidden_dim=512):
        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class Encoder(nn.Module):
    def __init__(self, input_dim=3601, latent_dim=144, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.ReLU()
                )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim=3601, latent_dim=144, hidden_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, input_dim),
                    nn.ReLU()
                )
    def forward(self, x):
        return self.decoder(x)

class Classifier(nn.Module):
    def __init__(self, encoder, input_dim=3601, latent_dim=144, hidden_dim=512, compounds=144, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.presence_head = nn.Linear(latent_dim, compounds)
        self.concentration_head = nn.Linear(latent_dim, compounds)

    def forward(self, x):
        encoded = self.encoder(x)

        pres_logits = self.presence_head(encoded)
        pres = torch.sigmoid(pres_logits)
        conc_logits = self.concentration_head(encoded)
        concs = torch.relu(conc_logits)

        gated_concs = concs * pres

        return pres, gated_concs

def train_autoencoder(train_data, ae_model, criterion, num_epochs=50, batch_size=32, lr=1e-3):
    # Autoencoders reconstruct themselves, so data is both input and target
    # Squeeze out the channel dimension [batch, 1, 3601] -> [batch, 3601] for MLP
    data = torch.tensor(train_data, dtype=torch.float32).view(len(train_data), -1)
    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr)
    criterion = criterion
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    losses = []
    ae_model.train()
    print(" =============== Launching AutoEncoder training ==============")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, target_data in dataloader:
            optimizer.zero_grad()
            reconstruction = ae_model(batch_data)
            loss = criterion(reconstruction, target_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"AE Epoch [{epoch+1}/{num_epochs}], MSE Loss: {avg_loss:.5f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), losses, label='AE Reconstruction Loss')
    plt.title('Autoencoder Pre-training')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    #plt.show()

def train_classifier(train_data, train_labels, classifier, weights_pth, detection_criterion, concentration_criterion, num_epochs=100, batch_size=32, lr=1e-3, hidden_dim=512):
    # Flatten input to [batch, 3601] for MLP
    print(f"Loading pre-trained encoder weights from {weights_pth} for classifier training")
    classifier.encoder.load_state_dict(torch.load(weights_pth))

    data = torch.tensor(train_data, dtype=torch.float32).view(len(train_data), -1)
    labels = torch.tensor(train_labels, dtype=torch.float32)

    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    
    # Only train parameters that require gradients (respects freeze_encoder)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    losses = []
    classifier.train()

    print(" ==================== Launching Classifier training =============")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (batch_data, batch_labels) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            pres_probs, gated_concs = classifier(batch_data)
            
            loss = compute_loss(pres_probs, gated_concs, batch_labels, detection_criterion, concentration_criterion, display=False)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Classifier Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

@torch.no_grad()
def test_classifier(test_data, test_labels, classifier, weights_pth, presence_threshold=0.5):
    print(f"Loading pre-trained encoder weights from {weights_pth} for testing")
    classifier.load_state_dict(torch.load(weights_pth))
    classifier.eval()
    sample_x = torch.tensor(test_data, dtype=torch.float32).view(len(test_data), -1)
    sample_y = np.array(test_labels)

    pres_probs, gated_concs = classifier(sample_x)
    
    presence_prob_np = pres_probs.cpu().numpy()
    final_pred = gated_concs.cpu().numpy()
    
    presence_labels = (sample_y > 0).astype(np.float32)
    pred_labels = (presence_prob_np > presence_threshold).astype(np.float32)

    # Clean up predicted concentrations below threshold to exactly 0
    final_pred = final_pred * pred_labels

    missed_mask = (presence_labels == 1) & (pred_labels == 0)
    hallucinated_mask = (presence_labels == 0) & (pred_labels == 1)

    misses_per_sample = missed_mask.sum(axis=1)
    hallucinations_per_sample = hallucinated_mask.sum(axis=1)

    total_samples = sample_x.shape[0]
    perfect_samples = ((misses_per_sample == 0) & (hallucinations_per_sample == 0)).sum()
    samples_with_misses = (misses_per_sample > 0).sum()
    samples_with_hallucinations = (hallucinations_per_sample > 0).sum()

    overall_mae = np.mean(np.abs(final_pred - sample_y))
    present_mask = presence_labels == 1
    present_mae = np.mean(np.abs(final_pred[present_mask] - sample_y[present_mask])) if np.any(present_mask) else 0.0

    print("=== COMPONENT IDENTIFICATION REPORT ===")
    print(f"Threshold used: {presence_threshold}")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Perfectly Identified Samples: {perfect_samples} ({(perfect_samples/total_samples)*100:.1f}%)")
    print(f"Samples missing ≥1 component: {samples_with_misses} ({(samples_with_misses/total_samples)*100:.1f}%)")
    print(f"Samples hallucinating ≥1 component: {samples_with_hallucinations} ({(samples_with_hallucinations/total_samples)*100:.1f}%)")
    
    print("\n=== AGGREGATE COMPONENT COUNTS ===")
    print(f"Total Missed (False Negatives): {missed_mask.sum()}")
    print(f"Total Hallucinated (False Positives): {hallucinated_mask.sum()}")
    print(f"Total Correct (True Positives): {((presence_labels == 1) & (pred_labels == 1)).sum()}")
    
    print("\n=== CONCENTRATION PERFORMANCE ===")
    print(f"Overall MAE:      {overall_mae:.5f}")
    print(f"Present-only MAE: {present_mae:.5f}")