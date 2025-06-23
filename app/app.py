import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import os
import tempfile
from scipy.io import wavfile
import io
import hashlib

# Set page config
st.set_page_config(
    page_title="InstruDetector",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .note-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Model Architecture
class SimpleAudioCNN(nn.Module):
    """
    Basic convolutional neural network for classifying mel spectrograms.
    Expects input of shape (batch, 1, n_mels, time).
    Returns logits of shape (batch, num_classes).
    """
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
        # Fully connected layers are initialized after the first forward pass
        self.fc1 = None
        self.fc2 = None
        self.n_classes = n_classes

    def forward(self, x):
        # Input: (batch, 1, n_mels, time)
        assert x.ndim == 4, f"Expected 4D input (batch, 1, n_mels, time), got {x.shape}"
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # Initialize fully connected layers on first pass
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
            self.fc2 = nn.Linear(128, self.n_classes).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_file_hash(path):
    """Return SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleAudioCNN(n_mels=128, n_classes=4).to(device)
    model_path = Path("outputs/run_20250609-150425/checkpoints/model_e17_acc81.8_20250609-194111.pt")
    state_dict = torch.load(model_path, map_location=device)
    # Dummy forward to initialize FC layers
    dummy = torch.zeros(1, 1, 128, 120).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy)
    # Now load weights (FC layers exist)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        st.sidebar.warning(f"Missing keys in state_dict: {missing}")
    if unexpected:
        st.sidebar.warning(f"Unexpected keys in state_dict: {unexpected}")
    st.sidebar.info(f"Model path: {model_path}")
    st.sidebar.info(f"Model hash: {get_file_hash(model_path)}")
    st.sidebar.info(f"Device: {device}")
    st.sidebar.info(f"Model arch: {model}")
    return model, device

# Class mapping (must match training exactly)
CLASS_MAP = {
    0: 'guitar',
    1: 'piano',
    2: 'mallet',
    3: 'string'
}

# Audio processing functions
def load_audio(file_path, target_sr=16000):
    """Load and resample audio file."""
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def detect_notes(y, sr, hop_length=512):
    """Detect note onsets in audio."""
    try:
        # For samples, just return the start time
        if len(y) / sr < 2.0:  # If audio is less than 2 seconds, treat as a sample
            return np.array([0.0])
            
        # For longer audio, use very conservative onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, 
            sr=sr,
            hop_length=hop_length,
            wait=1.0,  # Minimum 1 second between onsets
            pre_avg=0.5,  # Much larger windows
            post_avg=0.5,
            pre_max=0.5,
            post_max=0.5,
            delta=0.5,  # Much higher threshold
            units='time'
        )
        
        # If no onsets detected or only one onset, use the start of the audio
        if len(onset_frames) <= 1:
            return np.array([0.0])
            
        return onset_frames
    except Exception as e:
        st.warning("No notes detected in the audio file.")
        return np.array([0.0])

def create_mel_spectrogram(y, sr):
    """Create mel spectrogram with exact same parameters as training."""
    # Use same parameters as training
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_mels=128,
        fmin=20,
        fmax=8000,
        n_fft=2048,
        hop_length=512
    )
    # Convert to log scale (dB)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Clip to [-80, 0] dB range
    mel_spec = np.clip(mel_spec, -80.0, 0.0)
    
    # Ensure exactly 120 frames
    if mel_spec.shape[1] < 120:
        mel_spec = np.pad(
            mel_spec,
            ((0, 0), (0, 120 - mel_spec.shape[1])),
            mode='constant',
            constant_values=-80.0  # pad with silence
        )
    else:
        mel_spec = mel_spec[:, :120]
    
    return mel_spec

def standardize_mel_shape(mel, target_frames=120):
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode='constant', constant_values=-80.0)
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames]
    return mel

def preprocess_audio(y, sr, start_time, duration=1.5):
    try:
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        segment = y[start_sample:end_sample]
        # Deep debug: print segment stats and plot
        st.write("Segment stats:", {
            "mean": float(segment.mean()),
            "std": float(segment.std()),
            "min": float(segment.min()),
            "max": float(segment.max()),
            "energy": float(np.sum(segment ** 2))
        })
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(segment)
        ax.set_title('Extracted Audio Segment')
        st.pyplot(fig)
        mel_spec = create_mel_spectrogram(segment, sr)
        mel_spec = standardize_mel_shape(mel_spec)
        mel_spec = np.clip(mel_spec, -80.0, 0.0)
        st.write("Mel stats:", {
            "mean": float(mel_spec.mean()), "std": float(mel_spec.std()), "min": float(mel_spec.min()), "max": float(mel_spec.max())
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mel_spec, sr=sr, ax=ax)
        ax.set_title('Mel Spectrogram')
        st.pyplot(fig)
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        if mel_tensor.shape != (1, 1, 128, 120):
            st.warning(f"Adjusted tensor shape to (1, 1, 128, 120), got {mel_tensor.shape}")
        return mel_tensor
    except Exception as e:
        st.error(f"Error preprocessing audio: {str(e)}")
        return None

def test_model_sanity(model, device):
    """Test model on silence and noise in dB range."""
    # Create silence spectrogram
    silence = np.full((128, 120), -80.0)  # All silence
    silence_tensor = torch.FloatTensor(silence).unsqueeze(0).unsqueeze(0).to(device)
    
    # Create noise spectrogram in dB range
    noise = np.random.uniform(-80.0, 0.0, (128, 120))  # Random dB values
    noise_tensor = torch.FloatTensor(noise).unsqueeze(0).unsqueeze(0).to(device)
    
    # Test both
    with torch.no_grad():
        silence_logits = model(silence_tensor)
        noise_logits = model(noise_tensor)
        
        silence_probs = F.softmax(silence_logits, dim=1)[0]
        noise_probs = F.softmax(noise_logits, dim=1)[0]
        
        st.write("Silence test logits:", silence_logits.cpu().numpy())
        st.write("Silence test probs:", silence_probs.cpu().numpy())
        st.write("Noise test logits:", noise_logits.cpu().numpy())
        st.write("Noise test probs:", noise_probs.cpu().numpy())
        
        # Check if model predicts same class with high confidence
        max_silence = torch.max(silence_probs).item()
        max_noise = torch.max(noise_probs).item()
        
        if max_silence > 0.9 or max_noise > 0.9:
            st.error("⚠️ Model appears to be collapsed - predicting same class with high confidence on silence/noise!")
            return False
        return True

# Streamlit UI
def main():
    st.title("🎵 InstruDetector")
    st.markdown("### Instrument Classification from Audio")
    
    # Sidebar
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. 🎧 Upload a sample or use one of ours
    2. Select a note from the waveform
    3. View the prediction and confidence scores
    """)
    
    st.sidebar.markdown("### Sample Audio")
    st.sidebar.markdown("Try these sample files:")
    
    # Sample files
    sample_files = {
        "Guitar Sample": "data/filtered_train/guitar_acoustic_000-022-050.wav",
        "Piano Sample": "data/filtered_train/keyboard_electronic_062-074-127.wav",
        "Mallet Sample": "data/filtered_train/mallet_acoustic_063-069-100.wav",
        "String Sample": "data/filtered_train/string_acoustic_024-041-100.wav"
    }
    
    for label, path in sample_files.items():
        if st.sidebar.button(label):
            st.session_state['sample_file'] = path
    
    # Main content
    uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=['wav', 'mp3'])
    
    if uploaded_file is not None or 'sample_file' in st.session_state:
        # Process audio file
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
        else:
            audio_path = st.session_state['sample_file']
        
        # Load and process audio
        y, sr = load_audio(audio_path)
        if y is not None:
            # Display audio info
            duration = len(y) / sr
            st.markdown(f"**Audio Info:** Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
            
            # Detect notes
            onset_times = detect_notes(y, sr)
            
            # Create two columns for waveform and prediction
            col1, col2 = st.columns(2)
            
            with col1:
                # Display waveform with onsets
                fig, ax = plt.subplots(figsize=(8, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                if len(onset_times) > 0:
                    ax.vlines(onset_times, -1, 1, color='r', alpha=0.5, label='Note Onsets')
                ax.set_title('Waveform with Detected Notes')
                ax.legend()
                st.pyplot(fig)
            
            # Note selection
            st.markdown("### Select a Note")
            cols = st.columns(4)
            selected_note = None
            
            if len(onset_times) > 0:
                # For samples, just show one button
                if len(y) / sr < 2.0:
                    with cols[0]:
                        st.markdown("**Sample Note**")
                        if st.button("Select", key="select_sample"):
                            selected_note = 0.0
                        if st.button("Play", key="play_sample"):
                            st.audio(y, sample_rate=sr)
                else:
                    # For longer audio, show detected notes
                    for i, onset in enumerate(onset_times):
                        col_idx = i % 4
                        with cols[col_idx]:
                            st.markdown(f"**Note {i+1}:** {onset:.2f}s")
                            if st.button("Select", key=f"select_{i}"):
                                selected_note = onset
                            if st.button("Play", key=f"play_{i}"):
                                segment = y[int(onset * sr):int((onset + 0.5) * sr)]
                                st.audio(segment, sample_rate=sr)
            else:
                st.warning("No notes detected in the audio file. Try a different sample.")
            
            if selected_note is not None:
                # Load model
                model, device = load_model()
                
                # Run sanity check
                if not test_model_sanity(model, device):
                    st.warning("The model may not be working correctly. Please check the training process.")
                
                # Process selected note
                mel_tensor = preprocess_audio(y, sr, selected_note, duration=1.5)
                if mel_tensor is not None:
                    mel_tensor = mel_tensor.to(device)
                    
                    # Get prediction
                    with torch.no_grad():
                        # Debug: Print input tensor stats
                        st.write("Input tensor stats:")
                        st.write(f"Shape: {mel_tensor.shape}")
                        st.write(f"Min: {mel_tensor.min().item():.4f}")
                        st.write(f"Max: {mel_tensor.max().item():.4f}")
                        st.write(f"Mean: {mel_tensor.mean().item():.4f}")
                        st.write(f"Std: {mel_tensor.std().item():.4f}")
                        
                        # Get raw logits
                        logits = model(mel_tensor)
                        st.write("Raw logits:")
                        st.write(logits.cpu().numpy())
                        
                        # Get probabilities
                        probabilities = F.softmax(logits, dim=1)[0]
                        st.write("Softmax probabilities:")
                        st.write(probabilities.cpu().numpy())
                        
                        predicted_class = torch.argmax(probabilities).item()
                    
                    # Use consistent class mapping
                    class_names = [CLASS_MAP[i] for i in range(4)]
                    confidences = probabilities.cpu().numpy() * 100
                    
                    with col2:
                        # Display prediction
                        st.markdown("### Prediction")
                        st.markdown(f"""
                            <div class="prediction-box">
                                <h3>Predicted Instrument: {class_names[predicted_class].title()}</h3>
                                <p>Confidence: {confidences[predicted_class]:.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display top 3 predictions as bar chart
                        top_k = 3
                        top_indices = np.argsort(confidences)[-top_k:][::-1]
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = ax.barh(
                            [class_names[i].title() for i in top_indices],
                            [confidences[i] for i in top_indices],
                            color='skyblue'
                        )
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Top 3 Predictions')
                        
                        # Add confidence values to bars
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                                   f'{width:.1f}%', va='center')
                        
                        st.pyplot(fig)
                    
                    # Display mel spectrogram
                    st.markdown("### Mel Spectrogram")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    mel_spec = mel_tensor[0, 0].cpu().numpy()
                    librosa.display.specshow(mel_spec, sr=sr, ax=ax)
                    ax.set_title('Mel Spectrogram of Selected Note')
                    
                    # Add download button for spectrogram
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                    buf.seek(0)
                    st.download_button(
                        label="Download Spectrogram",
                        data=buf,
                        file_name="mel_spectrogram.png",
                        mime="image/png"
                    )
                    
                    st.pyplot(fig)

    # Add duration control in sidebar
    st.sidebar.markdown("### Audio Settings")
    duration = st.sidebar.slider("Analysis Duration (seconds)", 0.5, 2.0, 1.5, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Direct .npy Spectrogram Test")
    st.sidebar.write("CLASS_MAP:", CLASS_MAP)

    npy_file = st.sidebar.file_uploader("Upload .npy log-mel (128x120)", type=["npy"])
    skip_norm = st.sidebar.checkbox("Skip normalization (for pre-normalized .npy)", value=True)
    if npy_file is not None:
        mel = np.load(npy_file)
        mel = standardize_mel_shape(mel)
        st.write("Loaded .npy stats:", {
            "mean": float(mel.mean()), "std": float(mel.std()), "min": float(mel.min()), "max": float(mel.max())
        })
        if not skip_norm:
            mel = np.clip(mel, -80.0, 0.0)
            mel = (mel - (-20.0)) / 10.0
            st.info("Applied normalization.")
        else:
            st.info("Skipped normalization.")
        st.write("Final mel stats:", {
            "mean": float(mel.mean()), "std": float(mel.std()), "min": float(mel.min()), "max": float(mel.max())
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mel, sr=16000, ax=ax)
        ax.set_title('Uploaded .npy Mel Spectrogram')
        st.pyplot(fig)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
        model, device = load_model()
        with torch.no_grad():
            logits = model(mel_tensor.to(device))
            st.write("Raw logits:", logits.cpu().numpy())
            probs = F.softmax(logits, dim=1)[0]
            st.write("Softmax probabilities:", probs.cpu().numpy())
            pred = torch.argmax(probs).item()
            st.write(f"Predicted class: {CLASS_MAP[pred]}")

if __name__ == "__main__":
    main() 