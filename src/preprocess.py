import os
import json
import librosa
import numpy as np
import soundfile as sf  # Optional: For audio integrity check or loading

print("✅ preprocess.py loaded")



def make_mel_spectrogram(filepath, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Converts a .wav file to a Mel Spectrogram in decibel (dB) scale.

    Args:
        filepath (str): Path to the .wav file
        sr (int): Sampling rate (default: 16000)
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Number of samples between successive frames (default: 512)
        n_mels (int): Number of Mel bands to generate (default: 128)

    Returns:
        np.ndarray: Mel spectrogram in dB scale
    """
    # Basic integrity check using soundfile
    try:
        f = sf.SoundFile(filepath)
        assert f.samplerate == sr  # Optional: warn if unexpected sample rate
    except Exception as e:
        print(f"Warning: soundfile check failed for {filepath}: {e}")

    y, sr = librosa.load(filepath, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def process_folder(json_file, source_dir, target_dir):
    """
    Processes all .wav files listed in a filtered JSON file into mel spectrograms
    and saves them as .npy files.

    Args:
        json_file (str): Path to the filtered JSON file (contains only valid note IDs)
        source_dir (str): Directory with original .wav files
        target_dir (str): Directory to save mel spectrogram .npy files
    """
    with open(json_file) as f:
        metadata = json.load(f)

    # Create subdirectory only if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    for note_id in metadata:
        wav_path = os.path.join(source_dir, f"{note_id}.wav")
        mel_path = os.path.join(target_dir, f"{note_id}.npy")

        if os.path.exists(wav_path):
            mel = make_mel_spectrogram(wav_path)
            np.save(mel_path, mel)
        else:
            print(f"Missing: {wav_path}")



if __name__ == "__main__":
    pass