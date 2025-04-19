import json , shutil, os

"""
Filter NSynth dataset to keep only selected instrument families.

Each instrument in NSynth is labeled with exactly one instrument family.
This script filters for the following families and Index:

    - guitar (3)
    - keyboard (4)
    - mallet (percussive) (5)
    - string (includes violin, cello, etc.) (8)

It loads `nsynth-{split}.json` files from /data/json/, 
filters only the relevant samples, 
and writes `nsynth-{split}-filtered.json` in the same directory.

Then 

"""

def filter_data(json_data, allowed_families):
    """
    Filters NSynth metadata to include only specified instrument families.

    Args:
        json_data (dict): Full loaded JSON object from NSynth.
        allowed_families (list[str]): List of allowed `instrument_family_str` values.

    Returns:
        dict: Filtered metadata dictionary.
    """

    filtered = {}
    for key, entry in json_data.items():
        fam = entry["instrument_family_str"]
        if fam in allowed_families:
            filtered[key] = entry
    return filtered





allowed_families = ["guitar", "keyboard", "mallet", "string"]

paths = {
    "train": "data/json/nsynth-train.json",
    "valid": "data/json/nsynth-valid.json",
    "test":  "data/json/nsynth-test.json"
}

output_dir = "data/json"
os.makedirs(output_dir, exist_ok=True)

for name, path in paths.items():
    with open(path) as f:
        data = json.load(f)

    filtered = filter_data(data, allowed_families)

    output_path = os.path.join(output_dir, f"nsynth-{name}-filtered.json")
    with open(output_path, "w") as f_out:
        json.dump(filtered, f_out, indent=2)

    print(f"{name.capitalize()}: {len(filtered)} samples saved to {output_path}")



