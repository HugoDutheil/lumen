import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_dim=3601, hidden_dim=512, compounds=144, channels=1):
        super().__init__()
        self.trained_weights = None
        latent_dim = compounds-16
        self.block = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.2), 
            nn.Linear(in_features=hidden_dim, out_features=latent_dim), 
            nn.BatchNorm1d(num_features=latent_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.detection_head = nn.Linear(latent_dim, compounds)
        self.concentration_head = nn.Linear(latent_dim, compounds)

    def forward(self, x):
        extracted = self.block(x)
        probs_raw = self.detection_head(extracted)
        probs = torch.sigmoid(probs_raw)
        concs_raw = self.concentration_head(extracted)
        concs = torch.relu(concs_raw)

        gated_concs = probs * concs

        return probs, gated_concs

def compute_loss(presence_probs, gated_concentrations, weights, detection_criterion, concentration_criterion, display=False):
    presence_target = (weights > 0).float()

    detection_loss = detection_criterion(presence_probs, presence_target)

    pos_mask = presence_target.bool()  # (N, 144), elementwise
    if pos_mask.sum() > 0:
        conc_loss = concentration_criterion(gated_concentrations[pos_mask], weights[pos_mask])
    else:
        conc_loss = torch.tensor(0.0, device=presence_probs.device)

    total_loss = detection_loss + 1.0 * conc_loss

    if display:
        print(f"detection: {detection_loss.item():.4f} | concentration: {conc_loss.item():.4f}")

    return total_loss


@torch.no_grad()
def predict(model, x, presence_prob_cutoff=0.5):
    # Model now returns ready-to-use probabilities and gated concentrations
    presence_probs, gated_concentrations = model(x)
    
    # 1. Detection decision based directly on the model's output
    detected_mask = presence_probs > presence_prob_cutoff

    # 2. Clean up concentrations below the cutoff threshold to exact 0
    final_concentrations = gated_concentrations.masked_fill(~detected_mask, 0.0)

    # 3. Force the final detected concentrations to sum to 1.0 (since targets sum to 1)
    sums = final_concentrations.sum(dim=-1, keepdim=True)
    sums = torch.clamp(sums, min=1e-9) # Prevent division by zero
    final_concentrations = final_concentrations / sums

    # If no components were detected, return exact zeros across the board
    all_zero_mask = (detected_mask.sum(dim=-1, keepdim=True) == 0)
    final_concentrations = final_concentrations.masked_fill(all_zero_mask, 0.0)

    return detected_mask.float(), final_concentrations

def train_mlp(train_data, train_labels, model, detection_criterion, concentration_criterion, optimizer, num_epochs=100, batch_size=32):
    # Prepare data, convert to torch tensors and reshape for CNN input
    #print(f"The shape of the train dataset is {len(train_data[0])}")
    data = torch.tensor(train_data, dtype=torch.float32)#.unsqueeze(1)
    print(f"After Torch it is {data.shape}")
    labels = torch.tensor(train_labels, dtype=torch.float32)


    # Prepare DataLoader for batching
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    losses = []         
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (batch_data, batch_labels) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            #print(batch_data.dtype, next(model.parameters()).dtype)
            presence, concentrations = model(batch_data)
            display = False
            if epoch%100==0:
                display = True
            loss = compute_loss(presence, concentrations, batch_labels, detection_criterion, concentration_criterion, display)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
        scheduler.step(avg_epoch_loss)
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
    model.weights = model.block[0].weight.data.clone().detach().cpu().numpy()  # Store the weights of the first convolutional layer

def test_mlp(test_data, test_labels, model, weights_pth, hidden_dim=512, presence_threshold=0.5):
    # must match training-time args exactly
    print(f"The shape of the train dataset is {len(test_data[0])}")
    model = MLP(input_dim=len(test_data[0]), hidden_dim=hidden_dim)  
    model.load_state_dict(torch.load(weights_pth, map_location="cpu"))
    model.eval()
    
    with torch.no_grad():
        sample_x = torch.tensor(test_data, dtype=torch.float32)
        sample_y = test_labels

        # Using your predict() function which handles the masking and gating internally
        presence_prob_tensor, final_concentrations_tensor = predict(model, sample_x)
        
        presence_prob = presence_prob_tensor.cpu().numpy()
        final_pred = final_concentrations_tensor.cpu().numpy()
        
        # Ground truth and predicted presence masks
        presence_labels = (sample_y > 0).astype(np.float32)
        pred_labels = (presence_prob > presence_threshold).astype(np.float32)

        # ---------------------------------------------------------
        # Calculate hallucinations (False Positives) and misses (False Negatives)
        # ---------------------------------------------------------
        missed_mask = (presence_labels == 1) & (pred_labels == 0)
        hallucinated_mask = (presence_labels == 0) & (pred_labels == 1)

        misses_per_sample = missed_mask.sum(axis=1)
        hallucinations_per_sample = hallucinated_mask.sum(axis=1)

        # Calculate sample-level statistics
        total_samples = sample_x.shape[0]
        samples_with_misses = (misses_per_sample > 0).sum()
        nearly_correct_misses = (misses_per_sample == 1).sum()
        samples_with_hallucinations = (hallucinations_per_sample > 0).sum()
        nearly_correct_samples = (hallucinations_per_sample == 1).sum()
        perfect_samples = ((misses_per_sample == 0) & (hallucinations_per_sample == 0)).sum()

        # ---------------------------------------------------------
        # Concentration Head Metrics (Regression)
        # ---------------------------------------------------------
        overall_mae = np.mean(np.abs(final_pred - sample_y))
        present_mask = presence_labels == 1
        present_mae = np.mean(np.abs(final_pred[present_mask] - sample_y[present_mask])) if np.any(present_mask) else 0.0

        # ---------------------------------------------------------
        # Print Results
        # ---------------------------------------------------------
        print("=== COMPONENT IDENTIFICATION REPORT ===")
        print(f"Threshold used: {presence_threshold}")
        print(f"Total Samples Tested: {total_samples}")
        print(f"Perfectly Identified Samples: {perfect_samples} ({(perfect_samples/total_samples)*100:.1f}%)")
        print(f"Samples missing ≥1 component: {samples_with_misses} ({(samples_with_misses/total_samples)*100:.1f}%)")
        print(f"Samples missing 1 component: {nearly_correct_misses} ({(nearly_correct_misses/total_samples)*100:.1f}%)")
        print(f"Samples hallucinating ≥1 component: {samples_with_hallucinations} ({(samples_with_hallucinations/total_samples)*100:.1f}%)")
        print(f"Samples hallucinating 1 component: {nearly_correct_samples} ({(nearly_correct_samples/total_samples)*100:.1f}%)")
        
        print("\n=== AGGREGATE COMPONENT COUNTS ===")
        print(f"Total Missed Components (False Negatives): {missed_mask.sum()}")
        print(f"Total Hallucinated Components (False Positives): {hallucinated_mask.sum()}")
        print(f"Total Correctly Identified Components (True Positives): {((presence_labels == 1) & (pred_labels == 1)).sum()}")
        
        print("\n=== CONCENTRATION PERFORMANCE ===")
        print(f"Overall MAE:      {overall_mae:.5f}")
        print(f"Present-only MAE: {present_mae:.5f}")

        print("\n=== SAMPLE 0 DETAILS ===")
        print(f"GT Present count:   {presence_labels[0].sum()}")
        print(f"Pred Present count: {pred_labels[0].sum()}")
        print(f"Missed count:       {misses_per_sample[0]}")
        print(f"Hallucinated count: {hallucinations_per_sample[0]}")

        # ---------------------------------------------------------
        # Top-K Ranking Accuracy
        # ---------------------------------------------------------
        # Rank by actual concentrations
        gt_ranks = np.argsort(-sample_y, axis=1) 
        pred_ranks = np.argsort(-final_pred, axis=1)

        top1_match = (gt_ranks[:, 0] == pred_ranks[:, 0]).sum()
        top2_match = (gt_ranks[:, 1] == pred_ranks[:, 1]).sum()
        
        top3_set_match = 0
        for i in range(total_samples):
            if set(gt_ranks[i, :3]) == set(pred_ranks[i, :3]):
                top3_set_match += 1

        print("\n=== TOP-K RANKING ACCURACY ===")
        print(f"Top-1 Match (Correctly found the absolute highest component): {top1_match}/{total_samples} ({(top1_match/total_samples)*100:.1f}%)")
        print(f"Top-2 Exact Match (Correctly ranked the 2nd highest):       {top2_match}/{total_samples} ({(top2_match/total_samples)*100:.1f}%)")
        print(f"Top-3 Set Match (Found the top 3, regardless of order):     {top3_set_match}/{total_samples} ({(top3_set_match/total_samples)*100:.1f}%)")

    # ---------------------------------------------------------
    # Sample Examination
    # ---------------------------------------------------------
    sample = 10
    print(f"\nSample {sample} Ground Truth")
    print(sample_y[sample])
    print("============================================")
    print(f"Sample {sample} Predicted Presence Mask")
    print(pred_labels[sample])
    print("============================================")
    print(f"Sample {sample} Predicted Concentrations")
    print(final_pred[sample])