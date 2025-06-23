import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Model Architecture (must match training exactly)
class SimpleAudioCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        # Fixed FC layers (64 channels * 16 * 15 = 15360)
        self.fc1 = nn.Linear(64 * 16 * 15, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class mapping (must match training exactly)
CLASS_MAP = {
    0: 'guitar',
    1: 'piano',
    2: 'mallet',
    3: 'string'
}

def load_test_data(test_dir):
    """Load test spectrograms and their labels."""
    test_dir = Path(test_dir)
    files = list(test_dir.glob('*.npy'))
    
    specs = []
    labels = []
    
    # Print class distribution
    class_counts = {name: 0 for name in CLASS_MAP.values()}
    class_counts['keyboard'] = 0
    
    for file in tqdm(files, desc="Loading test data"):
        # Load spectrogram
        spec = np.load(file)
        spec = torch.FloatTensor(spec).unsqueeze(0)  # Add channel dimension
        
        # Extract label from filename
        instrument = file.stem.split('_')[0]
        class_counts[instrument] += 1
        
        # Map keyboard to piano
        if instrument == 'keyboard':
            instrument = 'piano'
            
        label = list(CLASS_MAP.values()).index(instrument)
        
        specs.append(spec)
        labels.append(label)
    
    # Print class distribution
    print("\nTest Set Class Distribution:")
    for instrument, count in class_counts.items():
        print(f"{instrument}: {count} samples")
    
    return torch.stack(specs), torch.tensor(labels)

def evaluate_model(model, test_specs, test_labels, device):
    """Evaluate model on test set and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for spec, label in tqdm(zip(test_specs, test_labels), total=len(test_specs), desc="Evaluating"):
            spec = spec.to(device)
            output = model(spec.unsqueeze(0))  # Add batch dimension
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            
            all_preds.append(pred.cpu().item())
            all_labels.append(label.item())
            all_probs.append(probs.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                 target_names=list(CLASS_MAP.values()),
                                 digits=3)
    
    # Calculate per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'class_accuracy': class_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': np.vstack(all_probs)
    }

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleAudioCNN(n_mels=128, n_classes=4).to(device)
    model_path = Path("outputs/run_20250609-150425/checkpoints/model_e17_acc81.8_20250609-194111.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load test data
    test_dir = "data/mel_spectrograms/test"
    test_specs, test_labels = load_test_data(test_dir)
    
    # Evaluate model
    results = evaluate_model(model, test_specs, test_labels, device)
    
    # Print results
    print("\n=== Model Evaluation Results ===")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print("\nPer-Class Accuracy:")
    for i, acc in enumerate(results['class_accuracy']):
        print(f"{CLASS_MAP[i].title()}: {acc:.3f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], list(CLASS_MAP.values()))
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Check for model collapse
    class_counts = np.bincount(results['predictions'])
    max_class = np.argmax(class_counts)
    max_class_ratio = class_counts[max_class] / len(results['predictions'])
    
    if max_class_ratio > 0.5:  # If any class is predicted more than 50% of the time
        print(f"\n⚠️ Warning: Possible model collapse detected!")
        print(f"Class '{CLASS_MAP[max_class]}' is predicted {max_class_ratio:.1%} of the time")
        
        # Check confidence distribution
        max_probs = np.max(results['probabilities'], axis=1)
        high_conf_ratio = np.mean(max_probs > 0.9)
        print(f"High confidence (>90%) predictions: {high_conf_ratio:.1%}")

if __name__ == "__main__":
    main() 