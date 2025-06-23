import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MelSpecDataset(Dataset):
    """
    Dataset for loading mel spectrograms and their labels from .npy files and a JSON metadata file.
    """
    def __init__(self, mel_dir, json_path, label_map=None, transform=None):
        """
        Args:
            mel_dir (str): Directory containing .npy spectrogram files.
            json_path (str): Path to JSON file with note_id to metadata mapping.
            label_map (dict, optional): Maps instrument family string to integer label.
            transform (callable, optional): Optional transform to apply to each spectrogram.
        """
        self.mel_dir = mel_dir
        self.transform = transform
        with open(json_path) as f:
            self.metadata = json.load(f)
        self.note_ids = list(self.metadata.keys())
        # If no label map is provided, build one from the metadata
        if label_map is None:
            fams = sorted({v['instrument_family_str'] for v in self.metadata.values()})
            self.label_map = {fam: i for i, fam in enumerate(fams)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.note_ids)

    def __getitem__(self, idx):
        note_id = self.note_ids[idx]
        mel_path = os.path.join(self.mel_dir, f"{note_id}.npy")
        mel = np.load(mel_path)
        # mel shape: (n_mels, time)
        if self.transform:
            mel = self.transform(mel)
        # Add channel dimension for CNN input (1, n_mels, time)
        mel = mel.astype(np.float32)
        mel = np.expand_dims(mel, axis=0)
        fam = self.metadata[note_id]['instrument_family_str']
        label = self.label_map[fam]
        return torch.from_numpy(mel), label

def test_dataset():
    """
    Quick test to load a small batch and print shapes/labels for sanity checking.
    """
    mel_dir = "data/mel_spectrograms/train"
    json_path = "data/json/nsynth-train-filtered.json"
    ds = MelSpecDataset(mel_dir, json_path)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    for xb, yb in dl:
        print(f"Batch mel shape: {xb.shape}")  # (batch, 1, n_mels, time)
        print(f"Batch labels: {yb}")
        break

# To run a quick test, uncomment below:
# test_dataset()
