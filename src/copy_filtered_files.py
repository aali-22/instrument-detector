"""
The script copies only the filtered .wav files into subfolders for
training, vaildation and testing datasets based on filtered files in filter_json.py

Filtered JSON files contains the note_ids of the desired instrument families
(e.g., guitar, keyboard, mallet, string), and each note_id corresponds to a .wav filename.

If the output folders (`data/filtered_*`) don't exist, they will be created automatically.
Missing .wav files will be logged, but won't break the script.


"""
import json, os, shutil

def copy_wav_files(json_path, source_dir, target_dir):
    """
    Copies .wav files listed in a filtered JSON metadata file to a new folder.

    Args:
        json_path (str): Path to filtered JSON file (e.g. "nsynth-train-filtered.json").
        source_dir (str): Directory containing original .wav files (e.g. "data/raw/train").
        target_dir (str): Directory to copy matching .wav files to (e.g. "data/filtered_train").

    Behavior:
    - If `target_dir` does not exist, it will be created (os.makedirs with exist_ok=True).
    - If a file listed in JSON is missing from `source_dir`, it will be logged but skipped.
    - File names are constructed using the note ID and the ".wav" extension.
    """


    
    # Load filtered metadata
    with open(json_path) as f:
        filtered = json.load(f)

    # Create target directory if it doesn't exist
    # exist_ok=True means it won't crash if the folder already exists
    os.makedirs(target_dir, exist_ok=True)

    # Loop through note_ids corresponding to a .wav filename
    for note_id in filtered:
        filename = f"{note_id}.wav"
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)

        # If source file exists, copy it to the new folder
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            # Log missing file but won't stop execution
            print(f"Missing file: {src}")


# Copy .wav files for all three dataset splits
copy_wav_files("data/json/nsynth-train-filtered.json",  "data/nsynth_raw/nsynth-train/audio",  "data/filtered_train")
copy_wav_files("data/json/nsynth-valid-filtered.json",  "data/nsynth_raw/nsynth-valid/audio",  "data/filtered_valid")
copy_wav_files("data/json/nsynth-test-filtered.json",   "data/nsynth_raw/nsynth-test/audio",   "data/filtered_test")
