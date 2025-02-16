import os
import sys

PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_DIR)
from constants import TRANSLATION_DATA_DIR, TRANSLATION_TRAIN_DIR, TRANSLATION_VAL_DIR, TRANSLATION_TEST_DIR


def read_data(source_file, target_file):
    """
    Reads parallel sentences from source and target files.

    Args:
        source_file (str): Path to the Vietnamese source file.
        target_file (str): Path to the English target file.

    Returns:
        tuple: Two lists containing English and Vietnamese sentences.
    """
    with open(source_file, "r", encoding="utf-8") as src_f, open(target_file, "r", encoding="utf-8") as tgt_f:
        source_lines = src_f.readlines()
        target_lines = tgt_f.readlines()

    assert len(source_lines) == len(target_lines), "Mismatch in number of lines!"
    return source_lines, target_lines


def split_data(source_lines, target_lines, train_ratio=0.8, val_ratio=0.1):
    """
    Splits data into train, validation, and test sets.

    Args:
        source_lines (list): List of source (Vietnamese) sentences.
        target_lines (list): List of target (English) sentences.
        train_ratio (float): Proportion of data to use for training (default=0.8).
        val_ratio (float): Proportion of data to use for validation (default=0.1).

    Returns:
        dict: Dictionary containing train, val, and test splits for both languages.
    """
    total_samples = len(source_lines)
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)

    return {
        "train": (source_lines[:train_end], target_lines[:train_end]),
        "val": (source_lines[train_end:val_end], target_lines[train_end:val_end]),
        "test": (source_lines[val_end:], target_lines[val_end:])
    }


def save_data(directory, prefix, source_data, target_data):
    """
    Saves source and target data into files.

    Args:
        directory (str): Target directory to save files.
        prefix (str): Prefix for file names (e.g., "train", "val", "test").
        source_data (list): List of source sentences.
        target_data (list): List of target sentences.
    """
    with open(os.path.join(directory, f"{prefix}.vi"), "w", encoding="utf-8") as src_f:
        src_f.writelines(source_data)

    with open(os.path.join(directory, f"{prefix}.en"), "w", encoding="utf-8") as tgt_f:
        tgt_f.writelines(target_data)


def split_and_save_data(source_file, target_file):
    """
    Splits the dataset and saves it into train, validation, and test folders.

    Args:
        base_dir (str): The base directory for the split dataset.
        source_file (str): Path to the original Vietnamese source file.
        target_file (str): Path to the original English target file.
    """
    # Step 1: Create directories
    os.makedirs(TRANSLATION_TRAIN_DIR, exist_ok=True)
    os.makedirs(TRANSLATION_VAL_DIR, exist_ok=True)
    os.makedirs(TRANSLATION_TEST_DIR, exist_ok=True)
    directories = {
        "train": TRANSLATION_TRAIN_DIR,
        "val": TRANSLATION_VAL_DIR,
        "test": TRANSLATION_TEST_DIR
    }

    # Step 2: Read original dataset
    source_lines, target_lines = read_data(source_file, target_file)

    # Step 3: Split the dataset
    splits = split_data(source_lines, target_lines)

    # Step 4: Save each split
    for split in ["train", "val", "test"]:
        save_data(directories[split], split, *splits[split])

    # Step 5: Delete original files
    # os.remove(source_file)
    # os.remove(target_file)

    print("Dataset successfully split!")
    print(f"Train: {len(splits['train'][0])} samples, Val: {len(splits['val'][0])} samples, Test: {len(splits['test'][0])} samples.")

# if __name__ == "__main__":
#     split_and_save_data(
#         source_file=os.path.join(TRANSLATION_DATA_DIR, "TED2020.en-vi.vi"),
#         target_file=os.path.join(TRANSLATION_DATA_DIR, "TED2020.en-vi.en")
#     )
