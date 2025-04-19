"""
Script to generate and save Mel Spectrograms (.npy) from filtered .wav files.

Assumes:
    - Corresponding .wav files are in filtered folders.
    - Mel spectrograms will be saved in /data/mel_spectrograms/{train, valid, test}.


"""

import os
from src.preprocess import process_folder

# Define paths for each split
configs = [
    {
        "json": "data/json/nsynth-train-filtered.json",
        "source": "data/filtered_train",
        "target": "data/mel_spectrograms/train"
    },
    {
        "json": "data/json/nsynth-valid-filtered.json",
        "source": "data/filtered_valid",
        "target": "data/mel_spectrograms/valid"
    },
    {
        "json": "data/json/nsynth-test-filtered.json",
        "source": "data/filtered_test",
        "target": "data/mel_spectrograms/test"
    }
]

if __name__ == "__main__":
    for cfg in configs:
        print(f"\n📁 Processing: {cfg['json']}")
        process_folder(cfg["json"], cfg["source"], cfg["target"])
        print(f"✅ Saved to {cfg['target']}")