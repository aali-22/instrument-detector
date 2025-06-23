# InstruDetector 🎵

A web application for instrument classification from audio samples. The application can detect individual notes in an audio file and classify them into four instrument categories: Guitar, Piano, Mallet, and String.

## Features

- Upload and process audio files (.wav or .mp3)
- Automatic note detection and segmentation
- Interactive note selection
- Real-time instrument classification
- Sample audio files for testing
- Beautiful visualization of waveforms and spectrograms

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/instrument-detector.git
cd instrument-detector
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download sample audio files:
Place sample audio files in the `app/static/samples` directory:
- `guitar_sample.wav`
- `piano_sample.wav`
- `mallet_sample.wav`
- `string_sample.wav`

## Usage

1. Start the Streamlit app:
```bash
cd app
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Use the application:
   - Upload an audio file or use the sample files
   - View the waveform with detected notes
   - Click on a note to classify it
   - See the prediction results and confidence scores

## Model Details

The application uses a CNN model trained on mel spectrograms with the following architecture:
- 3 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected layers for classification
- Input: Mel spectrograms (128 mel bands)
- Output: 4 instrument classes

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.22+
- Librosa 0.10+
- Other dependencies listed in requirements.txt

## License

MIT License

##  What is InstruDetector?

InstruDetector is a deep learning project that classifies isolated musical notes into instruments like guitar, piano, mallet, and strings using mel spectrograms and a CNN architecture.  
It transforms `.wav` files into spectrograms and learns to recognize the *sound signature / timbre* of each instrument.

---

##  Project Structure

- `notebooks/` – Demos, experiments, visualizations  
- `data/` – Contains filtered `.wav` files and generated mel spectrograms (`filtered_train`, `melspecs_train`, etc.)  
- `src/` – Core logic: preprocessing, training, evaluation  
- `outputs/` – Model checkpoints, logs, evaluation results


---

##  Example Use

```python
from src.preprocess import make_mel_spectrogram

spec = make_mel_spectrogram("data/filtered_train/guitar_001.wav")
