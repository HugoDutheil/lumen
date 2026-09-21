import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=256, compounds=144):
        super().__init__()
        self.trained_weights = None
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(), 
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(in_features=128*64, out_features=hidden_dim), # Can be dropped out if overfitting occurs
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            #nn.Linear(in_features=hidden_dim, out_features=compounds),
            #nn.LogSoftmax(dim=-1)  # Using LogSoftmax for numerical stability
        )

        self.detection_head = nn.Linear(in_features=hidden_dim, out_features=compounds)

        self.concentration_head = nn.Linear(in_features=hidden_dim, out_features=compounds)

    def forward(self, x):
        features = self.block(x)
        # Raw logits
        presence = self.detection_head(features)
        concentration = self.concentration_head(features)
        # Get probabilities and concentrations

        return presence, concentration #logits

class CNNSingleHead(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=256, compounds=144):
        super().__init__()
        self.trained_weights = None
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(), 
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(in_features=128*16, out_features=hidden_dim), # Can be dropped out if overfitting occurs
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_dim, out_features=compounds),
            nn.LogSoftmax(dim=-1)  # Using LogSoftmax for numerical stability
        )

    def forward(self, x):
        features = self.block(x)
        return features

def compute_loss(presence_logits, concentration_logits, weights, detection_criterion, concentration_criterion, display=False):
    presence_target = (weights > 0).float()
    
    #Detection loss across all 144 compounds (BCEWithLogitsLoss)
    detection_loss = detection_criterion(presence_logits, presence_target)

    #Masked Softmax for Concentration
    # Set logits of absent compounds to -1e9 so their exponential evaluates to 0.0
    pos_mask = presence_target.bool()
    masked_conc_logits = concentration_logits.masked_fill(~pos_mask, -1e9)
    
    # Softmax over masked logits ensures only present elements receive mass and sum to 1.0
    pred_normalized_conc = torch.softmax(masked_conc_logits, dim=-1)

    #Concentration loss strictly evaluated on present compounds
    if pos_mask.sum() > 0:
        concentration_loss = concentration_criterion(pred_normalized_conc[pos_mask], weights[pos_mask])
    else:
        concentration_loss = torch.tensor(0.0, device=presence_logits.device)

    total_loss = detection_loss + 10.0 * concentration_loss

    if display:
        print(f"detection: {detection_loss.item():.4f} | concentration: {concentration_loss.item():.4f}")

    return total_loss


def train(train_data, train_labels, model, detection_criterion, concentration_criterion, optimizer, num_epochs=100, batch_size=32):
    # Prepare data, convert to torch tensors and reshape for CNN input
    data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(train_labels, dtype=torch.float32)

    # Prepare DataLoader for batching
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    losses = []         
    model.train()
    for epoch in range(num_epochs):
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            #print(batch_data.dtype, next(model.parameters()).dtype)
            presence, concentrations = model(batch_data)
            display = False
            if epoch == 99:
                display = True
            loss = compute_loss(presence, concentrations, batch_labels, detection_criterion, concentration_criterion, display)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
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

def train_single_head(train_data, train_labels, model, criterion, optimizer, num_epochs=100, batch_size=32):
    # Prepare data, convert to torch tensors and reshape for CNN input
    data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(train_labels, dtype=torch.float32)

    # Prepare DataLoader for batching
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    losses = []         
    model.train()
    for epoch in range(num_epochs):
        #for batch_data, batch_labels in tqdm(dataloader):
        for i, (batch_data, batch_labels) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            #print(batch_data.dtype, next(model.parameters()).dtype)
            log_preds = model(batch_data)
            loss = criterion(log_preds, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    #plt.show()
    model.weights = model.block[0].weight.data.clone().detach().cpu().numpy()  # Store the weights of the first convolutional layer

def predict(model, x, threshold=0.1):
    model.eval()
    with torch.no_grad():
        presence_logits, concentration_logits = model(x)
        
        presence_prob = torch.sigmoid(presence_logits)
        detected_mask = presence_prob > threshold
        
        # Mask non-detected components to -inf so they receive 0 concentration
        masked_logits = concentration_logits.masked_fill(~detected_mask, -1e9)
        final_concentrations = torch.softmax(masked_logits, dim=-1)
        
        # Zero out samples where nothing was detected above threshold
        empty_samples = detected_mask.sum(dim=-1, keepdim=True) == 0
        final_concentrations = final_concentrations.masked_fill(empty_samples, 0.0)
        
        return presence_prob, final_concentrations

def test_single_head(test_data, test_labels, compounds, model, threshold=1e-1):

    model = CNNSingleHead(input_channels=1, hidden_dim=64, compounds=144)  # must match training-time args exactly
    model.load_state_dict(torch.load("model_weights_single_head.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        sample_x = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
        sample_y = test_labels

        # Model outputs LogSoftmax, so take exp() to get actual predicted percentages
        log_preds = model(sample_x)
        
        # Ground truth presence (1 if > 0, else 0)
        presence_labels = (sample_y > 0).astype(np.float32)
        
        # Predicted probabilities (removed .squeeze() to ensure 2D array: [samples, compounds])
        concentrationsm = torch.exp(log_preds).cpu().numpy()
        
        # Predicted presence based on your threshold
        pred_labels = (concentrationsm > threshold).astype(np.float32)

        # ---------------------------------------------------------
        # Calculate hallucinations (False Positives) and misses (False Negatives)
        # ---------------------------------------------------------
        # Missed: Ground truth says it's there (1), but prediction says no (0)
        missed_mask = (presence_labels == 1) & (pred_labels == 0)
        
        # Hallucinated: Ground truth says it's absent (0), but prediction says yes (1)
        hallucinated_mask = (presence_labels == 0) & (pred_labels == 1)

        # Count per sample (axis=1 sums across the compounds for each sample)
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
        # Print Results
        # ---------------------------------------------------------
        print("=== COMPONENT IDENTIFICATION REPORT ===")
        print(f"Threshold used: {threshold}")
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
        
        # Optional: Print detailed info for the first sample as a sanity check
        print("\n=== SAMPLE 0 DETAILS ===")
        print(f"GT Present count:   {presence_labels[0].sum()}")
        print(f"Pred Present count: {pred_labels[0].sum()}")
        print(f"Missed count:       {misses_per_sample[0]}")
        print(f"Hallucinated count: {hallucinations_per_sample[0]}")

        # Sort descending (-) to get the indices of the highest values
        gt_ranks = np.argsort(-sample_y, axis=1) 
        pred_ranks = np.argsort(-concentrationsm, axis=1)

        # Top 1 match: The highest predicted component is exactly the highest GT component
        top1_match = (gt_ranks[:, 0] == pred_ranks[:, 0]).sum()
        
        # Top 2 match: The 2nd highest predicted is exactly the 2nd highest GT
        # Note: This only makes sense for samples that actually have >= 2 components.
        # We'll calculate it just to see, but be aware of zero-ties in GT.
        top2_match = (gt_ranks[:, 1] == pred_ranks[:, 1]).sum()
        
        # Top 3 Unordered match: How often are the top 3 predicted components 
        # exactly the same set of components as the top 3 GT components?
        top3_set_match = 0
        for i in range(total_samples):
            if set(gt_ranks[i, :3]) == set(pred_ranks[i, :3]):
                top3_set_match += 1

        print("\n=== TOP-K RANKING ACCURACY ===")
        print(f"Top-1 Match (Correctly found the absolute highest component): {top1_match}/{total_samples} ({(top1_match/total_samples)*100:.1f}%)")
        print(f"Top-2 Exact Match (Correctly ranked the 2nd highest):       {top2_match}/{total_samples} ({(top2_match/total_samples)*100:.1f}%)")
        print(f"Top-3 Set Match (Found the top 3, regardless of order):     {top3_set_match}/{total_samples} ({(top3_set_match/total_samples)*100:.1f}%)")

def test_global(test_data, test_labels, compounds, model):

    model = CNN(input_channels=1, hidden_dim=64, compounds=144)  # must match training-time args exactly
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        sample_x = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
        sample_y = test_labels

        #pres_logits, conc_logits = model(sample_x)
        #probs = torch.sigmoid(pres_logits)[0].cpu().numpy()
        #conc_raw = conc_logits[0].cpu().numpy()

        #print(f"Max detection probability: {probs.max():.4f}")
        #print(f"Mean detection probability: {probs.mean():.4f}")
        #print(f"Number of compounds with prob > 0.5: {(probs > 0.5).sum()}")
        #print(f"Number of compounds with prob > 0.1: {(probs > 0.1).sum()}")
        #print(f"Top 5 highest probability indices: {np.argsort(probs)[::-1][:5]}")
        #print(f"Top 5 highest probabilities:       {np.sort(probs)[::-1][:5]}")

        # Model outputs LogSoftmax, so take exp() to get actual predicted percentages
        presence, concentrations = predict(model, sample_x)
        presence_labels = (sample_y>0).astype(np.float32)
        concentrationsm = (concentrations).numpy()
        presence_pred = (torch.sigmoid(presence).numpy() > 0.1).astype(np.float32)

        final_pred = presence_pred * concentrationsm

    #Get the percentage of predictions where components were hallucinated 
    true_positive = (presence_pred == 1) & (presence_labels == 1)
    false_positive = (presence_pred == 1) & (presence_labels == 0)
    true_negative = (presence_pred == 0) & (presence_labels == 0)
    false_negative = (presence_pred == 0) & (presence_labels == 1)

    # 3. Concentration Head Metrics (Regression)
    overall_mae = np.mean(np.abs(final_pred - sample_y))
    
    # MAE only on elements that are actually present
    present_mask = presence_labels == 1
    present_mae = np.mean(np.abs(final_pred[present_mask] - sample_y[present_mask])) if np.any(present_mask) else 0.0

    print(f"=== Concentration Performance ===")
    print(f"Overall MAE:      {overall_mae:.5f}")
    print(f"Present-only MAE: {present_mae:.5f}\n")
    


    # Get the top 5 highest predicted components
    print(sample_y[1500])
    print("--------------------")
    print(presence.numpy()[1500])
    print("--------------------")
    print(final_pred[1500])
"""     top_5_idx = np.argsort(concentrationsm, axis=1)[::-1][:, :5]
    sample_y_np = sample_y.numpy() if torch.is_tensor(sample_y) else sample_y
    top_5_true = np.take_along_axis(sample_y_np, top_5_idx, axis=1)
    errors = np.mean(np.abs(concentrationsm[top_5_idx] - top_5_true), axis=0)
    print(top_5_idx[:3])
    print(top_5_true[:3])
    print(f"Errors for the top 5 predicted components: {errors}")

    top_5_true = np.argsort(concentrationsm, axis=1)[::-1][:, :5]
    sample_y_np = sample_y.numpy() if torch.is_tensor(sample_y) else sample_y
    top_5_pred = np.take_along_axis(sample_y_np, top_5_true, axis=1)
    errors = np.mean(np.abs(top_5_pred - top_5_true), axis=0)
    print(top_5_pred[:3])
    print(top_5_true[:3])
    print(f"Errors for the top 5 predicted components: {errors}")

    #Get the percentage of predictions where components were hallucinated 
    true_positive = (presence_pred == 1) & (presence_labels == 1)
    false_positive = (presence_pred == 1) & (presence_labels == 0)
    true_negative = (presence_pred == 0) & (presence_labels == 0)
    false_negative = (presence_pred == 0) & (presence_labels == 1)

    hallucinated = false_positive.sum(axis=0)
    num_missed = false_negative.sum(axis=0)
    print(f"Hallucinations per component: {hallucinated}")
    print(f"Missed per component: {num_missed}") """

