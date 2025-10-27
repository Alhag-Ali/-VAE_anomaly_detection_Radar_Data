import os
import sys

# Insert the parent directory (VAE_anomaly_detection) into module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataset import radar_dataset
from torch.utils.data import TensorDataset, DataLoader
from model.VAE import VAEAnomalyTabular

# For ROC AUC and plotting
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np

# For plotting confusion matrix
import seaborn as sns

#load X_test
data_dir_test = r"Link\to\testdataset"

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

train_dataset, val_dataset, test_dataset = radar_dataset(image_size=128)
test_dloader = DataLoader(test_dataset, batch_size=16)
train_dloader = DataLoader(train_dataset, batch_size=16) # Add DataLoader for training set

# Get the absolute path to the VAE_anomaly_detection directory
VAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Construct the checkpoint path relative to VAE_ROOT
checkpoint_path = os.path.join(VAE_ROOT, 'saved_models', '2025-10-10_18-39-37', 'epoch=14-val', 'loss=-687566.00.ckpt')
print(f"Attempting to load checkpoint from: {checkpoint_path}")

# Load the model using the correct method
model = VAEAnomalyTabular.load_from_checkpoint(checkpoint_path)
model.eval() # Set model to evaluation mode

all_test_labels = []
all_test_probabilities = []
all_test_latent_mus = [] # To store latent_mu values

# --- Test Set Analysis ---
print("\n--- Analyzing Test Set ---")
with torch.no_grad(): # Disable gradient calculations for inference
    for batch_idx, (X_batch, y_batch) in enumerate(test_dloader):
        # We only collect probabilities and true labels for ROC AUC
        pred_results = model.predict(X_batch) # Get full prediction results
        batch_probabilities = model.reconstructed_probability(X_batch)
        
        all_test_labels.append(y_batch)
        all_test_probabilities.append(batch_probabilities)
        all_test_latent_mus.append(pred_results['latent_mu'])

# Concatenate all test results
all_test_labels = torch.cat(all_test_labels)
all_test_probabilities = torch.cat(all_test_probabilities)
all_test_latent_mus = torch.cat(all_test_latent_mus)

# Convert to numpy for sklearn and plotting
y_true = all_test_labels.cpu().numpy()
y_scores = all_test_probabilities.cpu().numpy()
latent_mus_np = all_test_latent_mus.cpu().numpy()

# For ROC curve, higher scores should indicate positive class (anomaly). 
# Since lower probability means anomaly, we'll negate the scores for roc_curve.
# This means: anomaly (label 1) -> low probability -> high -probability score
y_scores_for_roc = -y_scores

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores_for_roc, pos_label=1)
roc_auc = auc(fpr, tpr)

print(f"\n--- ROC AUC Score: {roc_auc:.4f} ---")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Analyze detected anomalies vs. alpha thresholds ---
# Define a range of alpha thresholds to test
# Since y_scores are probabilities, lower values are anomalous.
# Thresholds should be chosen from the range of y_scores.
min_score = y_scores.min()
max_score = y_scores.max()
alphas = np.linspace(min_score * 0.9, max_score * 1.1, 100) # Extend range slightly

detected_anomalies_counts = []
for alpha_val in alphas:
    # Anomalies are where probability < alpha_val
    num_detected = (y_scores < alpha_val).sum()
    detected_anomalies_counts.append(num_detected)

plt.figure(figsize=(8, 6))
plt.plot(alphas, detected_anomalies_counts, marker='o', linestyle='-', markersize=4)
plt.xlabel('Alpha Threshold (Lower Probability = Anomaly)')
plt.ylabel('Number of Detected Anomalies')
plt.title('Detected Anomalies vs. Alpha Threshold')
plt.grid(True)
plt.show()

# --- Classification Metrics for a chosen alpha threshold ---
print(f"\n--- Classification Metrics ---")
# Use the alpha value you previously experimented with, or choose a new one based on ROC/threshold plot
chosen_alpha = 9.0

y_pred = (y_scores < chosen_alpha).astype(int) # Convert boolean predictions to 0/1

precision = precision_score(y_true, y_pred, pos_label=1)
recall = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]) # 0 for normal, 1 for anomaly

print(f"Chosen Alpha Threshold: {chosen_alpha:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, classes=['Normal', 'Anomaly'], title='Confusion Matrix at Alpha {chosen_alpha:.4f}')

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

# --- Latent Space Visualization ---
print("\n--- Latent Space Visualization ---")

# Separate normal and anomaly latent_mu
latent_mu_normal = latent_mus_np[y_true == 0]
latent_mu_anomaly = latent_mus_np[y_true == 1]

# Plotting the first two latent dimensions
plt.figure(figsize=(10, 8))
plt.scatter(latent_mu_normal[:, 0], latent_mu_normal[:, 1], alpha=0.5, label='Normal', color='blue')
plt.scatter(latent_mu_anomaly[:, 0], latent_mu_anomaly[:, 1], alpha=0.8, label='Anomaly', color='red')

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Distribution (First 2 Dimensions)')
plt.legend()
plt.grid(True)
plt.show()


# --- Training Set Analysis (for comparison) ---
print("\n--- Analyzing Training Set ---")
all_train_probabilities = []
with torch.no_grad():
    for batch_idx, X_batch in enumerate(train_dloader):
        batch_probabilities = model.reconstructed_probability(X_batch)
        all_train_probabilities.append(batch_probabilities)

all_train_probabilities = torch.cat(all_train_probabilities)

print(f"\n--- Reconstructed Probability Analysis (Training Set) ---")
print(f"Min Probability: {all_train_probabilities.min().item():.4f}")
print(f"Max Probability: {all_train_probabilities.max().item():.4f}")
print(f"Mean Probability: {all_train_probabilities.mean().item():.4f}")
print(f"Median Probability: {all_train_probabilities.median().item():.4f}")

# --- Erweiterte Analyse: Verteilung der Rekonstruktionsfehler ---
print("\n--- Detaillierte Verteilungsanalyse ---")

# Separiere normale und anomalie Testdaten
normal_test_scores = y_scores[y_true == 0]
anomaly_test_scores = y_scores[y_true == 1]

print(f"\n=== TEST SET STATISTIKEN ===")
print(f"Normale Daten (n={len(normal_test_scores)}):")
print(f"  Min: {normal_test_scores.min():.6f}")
print(f"  Max: {normal_test_scores.max():.6f}")
print(f"  Mean: {normal_test_scores.mean():.6f}")
print(f"  Median: {np.median(normal_test_scores):.6f}")
print(f"  Std: {normal_test_scores.std():.6f}")

print(f"\nAnomalie Daten (n={len(anomaly_test_scores)}):")
print(f"  Min: {anomaly_test_scores.min():.6f}")
print(f"  Max: {anomaly_test_scores.max():.6f}")
print(f"  Mean: {anomaly_test_scores.mean():.6f}")
print(f"  Median: {np.median(anomaly_test_scores):.6f}")
print(f"  Std: {anomaly_test_scores.std():.6f}")

# Training Set (nur normale Daten)
train_scores = all_train_probabilities.cpu().numpy()
print(f"\n=== TRAINING SET STATISTIKEN (nur normale Daten) ===")
print(f"Training normale Daten (n={len(train_scores)}):")
print(f"  Min: {train_scores.min():.6f}")
print(f"  Max: {train_scores.max():.6f}")
print(f"  Mean: {train_scores.mean():.6f}")
print(f"  Median: {np.median(train_scores):.6f}")
print(f"  Std: {train_scores.std():.6f}")

# Visualisierung der Verteilungen
plt.figure(figsize=(15, 5))

# Subplot 1: Histogramm der Verteilungen
plt.subplot(1, 3, 1)
plt.hist(train_scores, bins=50, alpha=0.7, label='Training (normal)', color='blue', density=True)
plt.hist(normal_test_scores, bins=50, alpha=0.7, label='Test (normal)', color='green', density=True)
plt.hist(anomaly_test_scores, bins=50, alpha=0.7, label='Test (anomaly)', color='red', density=True)
plt.xlabel('Reconstruction Probability')
plt.ylabel('Density')
plt.title('Verteilung der Rekonstruktionswahrscheinlichkeiten')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Boxplot Vergleich
plt.subplot(1, 3, 2)
data_to_plot = [train_scores, normal_test_scores, anomaly_test_scores]
labels = ['Training\n(normal)', 'Test\n(normal)', 'Test\n(anomaly)']
plt.boxplot(data_to_plot, labels=labels)
plt.ylabel('Reconstruction Probability')
plt.title('Boxplot Vergleich der Verteilungen')
plt.grid(True, alpha=0.3)

# Subplot 3: Overlap Analyse
plt.subplot(1, 3, 3)
# Zeige die Überlappung zwischen normalen und anomalen Daten
plt.hist(normal_test_scores, bins=30, alpha=0.6, label='Normal', color='green', density=True)
plt.hist(anomaly_test_scores, bins=30, alpha=0.6, label='Anomaly', color='red', density=True)
plt.axvline(x=chosen_alpha, color='black', linestyle='--', label=f'Threshold = {chosen_alpha}')
plt.xlabel('Reconstruction Probability')
plt.ylabel('Density')
plt.title('Normal vs Anomaly Verteilung')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Optimaler Threshold basierend auf Verteilungen ---
print(f"\n=== THRESHOLD OPTIMIERUNG ===")

# Verschiedene Threshold-Strategien
thresholds_to_test = [
    np.percentile(train_scores, 95),  # 95% der Trainingsdaten
    np.percentile(train_scores, 99),  # 99% der Trainingsdaten
    np.percentile(train_scores, 99.5),  # 99.5% der Trainingsdaten
    np.mean(train_scores) - 2 * np.std(train_scores),  # 2 Standardabweichungen unter dem Mittelwert
    np.mean(train_scores) - 3 * np.std(train_scores),  # 3 Standardabweichungen unter dem Mittelwert
]

print("Verschiedene Threshold-Strategien:")
for i, thresh in enumerate(thresholds_to_test):
    y_pred_thresh = (y_scores < thresh).astype(int)
    precision = precision_score(y_true, y_pred_thresh, pos_label=1)
    recall = recall_score(y_true, y_pred_thresh, pos_label=1)
    f1 = f1_score(y_true, y_pred_thresh, pos_label=1)
    
    print(f"  Strategie {i+1}: Threshold = {thresh:.6f}")
    print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

print(f"\nAktueller gewählter Threshold: {chosen_alpha}")
print(f"Anzahl erkannte Anomalien: {(y_scores < chosen_alpha).sum()}")
print(f"Anzahl tatsächliche Anomalien: {y_true.sum()}")

# --- Visuelle Rekonstruktionsanalyse ---
print(f"\n=== REKONSTRUKTIONSVERGLEICH ===")

def visualize_reconstruction(model, test_dloader, num_examples=3):
    """
    Zeigt Original vs. Rekonstruktion für normale und anomalie Beispiele
    """
    model.eval()
    
    # Sammle Beispiele
    normal_examples = []
    anomaly_examples = []
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_dloader):
            # Normale Beispiele sammeln
            normal_mask = (y_batch == 0)
            if normal_mask.any() and len(normal_examples) < num_examples:
                normal_indices = torch.where(normal_mask)[0]
                for idx in normal_indices[:num_examples - len(normal_examples)]:
                    normal_examples.append({
                        'original': X_batch[idx],
                        'label': y_batch[idx].item(),
                        'batch_idx': batch_idx,
                        'sample_idx': idx.item()
                    })
            
            # Anomalie Beispiele sammeln
            anomaly_mask = (y_batch == 1)
            if anomaly_mask.any() and len(anomaly_examples) < num_examples:
                anomaly_indices = torch.where(anomaly_mask)[0]
                for idx in anomaly_indices[:num_examples - len(anomaly_examples)]:
                    anomaly_examples.append({
                        'original': X_batch[idx],
                        'label': y_batch[idx].item(),
                        'batch_idx': batch_idx,
                        'sample_idx': idx.item()
                    })
            
            if len(normal_examples) >= num_examples and len(anomaly_examples) >= num_examples:
                break
    
    # Rekonstruktionen berechnen
    def get_reconstruction(original_img):
        original_img = original_img.unsqueeze(0)  # Add batch dimension
        pred_result = model.predict(original_img)
        recon_mu = pred_result['recon_mu'].mean(dim=0)  # Average over L samples
        return recon_mu.squeeze(0)  # Remove batch dimension
    
    # Normale Beispiele verarbeiten
    for i, example in enumerate(normal_examples):
        reconstruction = get_reconstruction(example['original'])
        recon_prob = model.reconstructed_probability(example['original'].unsqueeze(0)).item()
        example['reconstruction'] = reconstruction
        example['recon_prob'] = recon_prob
    
    # Anomalie Beispiele verarbeiten
    for i, example in enumerate(anomaly_examples):
        reconstruction = get_reconstruction(example['original'])
        recon_prob = model.reconstructed_probability(example['original'].unsqueeze(0)).item()
        example['reconstruction'] = reconstruction
        example['recon_prob'] = recon_prob
    
    # Visualisierung
    fig, axes = plt.subplots(4, num_examples, figsize=(4*num_examples, 12))
    if num_examples == 1:
        axes = axes.reshape(-1, 1)
    
    # Normale Beispiele
    for i, example in enumerate(normal_examples):
        # Original
        axes[0, i].imshow(example['original'].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original (Normal)\nBatch {example["batch_idx"]}, Sample {example["sample_idx"]}')
        axes[0, i].axis('off')
        
        # Rekonstruktion
        axes[1, i].imshow(example['reconstruction'].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Rekonstruktion\nProb: {example["recon_prob"]:.6f}')
        axes[1, i].axis('off')
        
        # Differenz
        diff = torch.abs(example['original'] - example['reconstruction'])
        axes[2, i].imshow(diff.squeeze().detach().cpu().numpy(), cmap='hot')
        axes[2, i].set_title(f'Fehler (|Original - Rekon|)\nMean: {diff.mean().detach().item():.6f}')
        axes[2, i].axis('off')
        
        # Histogramm der Rekonstruktionswahrscheinlichkeit
        axes[3, i].hist([example['recon_prob']], bins=1, alpha=0.7, color='green')
        axes[3, i].set_title(f'Rekonstruktions-\nwahrscheinlichkeit')
        axes[3, i].set_xlabel('Probability')
        axes[3, i].set_ylabel('Count')
        axes[3, i].axvline(x=chosen_alpha, color='red', linestyle='--', label=f'Threshold={chosen_alpha}')
        axes[3, i].legend()
    
    # Anomalie Beispiele
    for i, example in enumerate(anomaly_examples):
        # Original
        axes[0, i].imshow(example['original'].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original (Anomalie)\nBatch {example["batch_idx"]}, Sample {example["sample_idx"]}')
        axes[0, i].axis('off')
        
        # Rekonstruktion
        axes[1, i].imshow(example['reconstruction'].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Rekonstruktion\nProb: {example["recon_prob"]:.6f}')
        axes[1, i].axis('off')
        
        # Differenz
        diff = torch.abs(example['original'] - example['reconstruction'])
        axes[2, i].imshow(diff.squeeze().detach().cpu().numpy(), cmap='hot')
        axes[2, i].set_title(f'Fehler (|Original - Rekon|)\nMean: {diff.mean().detach().item():.6f}')
        axes[2, i].axis('off')
        
        # Histogramm der Rekonstruktionswahrscheinlichkeit
        axes[3, i].hist([example['recon_prob']], bins=1, alpha=0.7, color='red')
        axes[3, i].set_title(f'Rekonstruktions-\nwahrscheinlichkeit')
        axes[3, i].set_xlabel('Probability')
        axes[3, i].set_ylabel('Count')
        axes[3, i].axvline(x=chosen_alpha, color='red', linestyle='--', label=f'Threshold={chosen_alpha}')
        axes[3, i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Zusammenfassung
    print(f"\n=== REKONSTRUKTIONSZUSAMMENFASSUNG ===")
    print(f"Normale Beispiele:")
    for i, example in enumerate(normal_examples):
        print(f"  Beispiel {i+1}: Rekonstruktionswahrscheinlichkeit = {example['recon_prob']:.6f}")
        print(f"    Mean Fehler: {torch.abs(example['original'] - example['reconstruction']).mean().detach().item():.6f}")
    
    print(f"\nAnomalie Beispiele:")
    for i, example in enumerate(anomaly_examples):
        print(f"  Beispiel {i+1}: Rekonstruktionswahrscheinlichkeit = {example['recon_prob']:.6f}")
        print(f"    Mean Fehler: {torch.abs(example['original'] - example['reconstruction']).mean().detach().item():.6f}")

# Führe die Rekonstruktionsanalyse durch
visualize_reconstruction(model, test_dloader, num_examples=3)

# --- Präsentationsfolien für Rekonstruktionsvergleich ---
print(f"\n=== CREATING PRESENTATION SLIDES ===")

def create_presentation_slides(model, test_dloader):
    """
    Erstellt Präsentationsfolien für Rekonstruktionsvergleich
    """
    model.eval()
    
    # Sammle je ein Beispiel für normal und anomalie
    normal_example = None
    anomaly_example = None
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_dloader):
            # Normales Beispiel finden
            if normal_example is None:
                normal_mask = (y_batch == 0)
                if normal_mask.any():
                    normal_idx = torch.where(normal_mask)[0][0]
                    normal_example = {
                        'original': X_batch[normal_idx],
                        'label': y_batch[normal_idx].item()
                    }
            
            # Anomalie Beispiel finden
            if anomaly_example is None:
                anomaly_mask = (y_batch == 1)
                if anomaly_mask.any():
                    anomaly_idx = torch.where(anomaly_mask)[0][0]
                    anomaly_example = {
                        'original': X_batch[anomaly_idx],
                        'label': y_batch[anomaly_idx].item()
                    }
            
            if normal_example is not None and anomaly_example is not None:
                break
    
    if normal_example is None or anomaly_example is None:
        print("Error: Could not find both normal and anomaly examples!")
        return
    
    # Rekonstruktionen berechnen
    def get_reconstruction(original_img):
        original_img = original_img.unsqueeze(0)
        pred_result = model.predict(original_img)
        recon_mu = pred_result['recon_mu'].mean(dim=0)
        return recon_mu.squeeze(0)
    
    # Normales Beispiel verarbeiten
    normal_recon = get_reconstruction(normal_example['original'])
    normal_prob = model.reconstructed_probability(normal_example['original'].unsqueeze(0)).item()
    normal_error = torch.abs(normal_example['original'] - normal_recon).mean().detach().item()
    
    # Anomalie Beispiel verarbeiten
    anomaly_recon = get_reconstruction(anomaly_example['original'])
    anomaly_prob = model.reconstructed_probability(anomaly_example['original'].unsqueeze(0)).item()
    anomaly_error = torch.abs(anomaly_example['original'] - anomaly_recon).mean().detach().item()
    
    # Folie 1: Rekonstruktion normaler Aktivitäten
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig1.suptitle('Reconstruction of Normal Activities', fontsize=16, fontweight='bold')
    
    # Original (Normal)
    ax1.imshow(normal_example['original'].squeeze().detach().cpu().numpy(), cmap='gray')
    ax1.set_title('Original Image\n(Normal Activity)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Rekonstruktion (Normal)
    ax2.imshow(normal_recon.squeeze().detach().cpu().numpy(), cmap='gray')
    ax2.set_title('VAE Reconstruction\n(Normal Activity)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Rekonstruktionsfehler hinzufügen
    fig1.text(0.5, 0.02, f'Reconstruction Error: {normal_error:.1f}', 
              ha='center', fontsize=12, fontweight='bold', 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    
    # Folie 2: Erkennung einer Anomalie (Sturz)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig2.suptitle('Anomaly Detection (Fall)', fontsize=16, fontweight='bold')
    
    # Original (Anomalie)
    ax1.imshow(anomaly_example['original'].squeeze().detach().cpu().numpy(), cmap='gray')
    ax1.set_title('Original Image\n(Fall Activity)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Rekonstruktion (Anomalie)
    ax2.imshow(anomaly_recon.squeeze().detach().cpu().numpy(), cmap='gray')
    ax2.set_title('VAE Reconstruction\n(Fall Activity)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Rekonstruktionsfehler hinzufügen
    fig2.text(0.5, 0.02, f'Reconstruction Error: {anomaly_error:.1f}', 
              ha='center', fontsize=12, fontweight='bold',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    
    # Zusätzliche Vergleichsfolie
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle('VAE Anomaly Detection: Normal vs Anomaly Comparison', fontsize=16, fontweight='bold')
    
    # Normal Original
    axes[0, 0].imshow(normal_example['original'].squeeze().detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Normal Activity\n(Original)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Normal Reconstruction
    axes[0, 1].imshow(normal_recon.squeeze().detach().cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Normal Activity\n(Reconstruction)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Anomaly Original
    axes[1, 0].imshow(anomaly_example['original'].squeeze().detach().cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Fall Activity\n(Original)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Anomaly Reconstruction
    axes[1, 1].imshow(anomaly_recon.squeeze().detach().cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Fall Activity\n(Reconstruction)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Fehler-Statistiken hinzufügen
    fig3.text(0.25, 0.02, f'Normal Error: {normal_error:.1f}', 
              ha='center', fontsize=11, fontweight='bold',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    fig3.text(0.75, 0.02, f'Anomaly Error: {anomaly_error:.1f}', 
              ha='center', fontsize=11, fontweight='bold',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    
    # Zusammenfassung
    print(f"\n=== PRESENTATION SUMMARY ===")
    print(f"Normal Activity:")
    print(f"  - Reconstruction Error: {normal_error:.1f}")
    print(f"  - Reconstruction Probability: {normal_prob:.6f}")
    print(f"  - Status: {'GOOD RECONSTRUCTION' if normal_error < 10 else 'POOR RECONSTRUCTION'}")
    
    print(f"\nFall Activity:")
    print(f"  - Reconstruction Error: {anomaly_error:.1f}")
    print(f"  - Reconstruction Probability: {anomaly_prob:.6f}")
    print(f"  - Status: {'DETECTED AS ANOMALY' if anomaly_error > normal_error * 1.5 else 'NOT CLEARLY DETECTED'}")

# Erstelle die Präsentationsfolien
create_presentation_slides(model, test_dloader)
