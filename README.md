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
