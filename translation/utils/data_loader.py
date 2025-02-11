import os
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from translation_dataset import TranslationDataset
from tokenization import get_token_transforms

PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_DIR)
from constants import PAD_IDX, BOS_IDX, EOS_IDX, SRC_LANGUAGE, TGT_LANGUAGE, MAX_LEN


def get_text_transforms(vocab_transform):
    """
    Returns text transformation pipelines for both source and target languages.

    Args:
        vocab_transform (dict): Vocabulary dictionary for both languages.

    Returns:
        dict: Dictionary containing transformation functions for each language.
    """
    token_transform = get_token_transforms()

    def sequential_transforms(*transforms):
        """Helper function to apply multiple sequential transformations."""
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    def tensor_transform(token_ids):
        """Adds BOS/EOS tokens and converts token IDs to a tensor."""
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    return {
        SRC_LANGUAGE: sequential_transforms(
            token_transform[SRC_LANGUAGE],  # Tokenization
            vocab_transform[SRC_LANGUAGE],  # Numericalization
            tensor_transform  # Add BOS/EOS and convert to tensor
        ),
        TGT_LANGUAGE: sequential_transforms(
            token_transform[TGT_LANGUAGE],
            vocab_transform[TGT_LANGUAGE],
            tensor_transform
        )
    }

def truncate(sequence):
    """Truncates sequences longer than MAX_LEN."""
    return sequence[:MAX_LEN] if sequence.size(0) > MAX_LEN else sequence

def collate_fn(batch, text_transform):
    """
    Collates a batch of samples into padded tensors.

    Args:
        batch (list): Batch of data samples.
        text_transform (dict): Dictionary of transformation functions.

    Returns:
        tuple: Padded and truncated source and target tensors.
    """
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_sample = text_transform[SRC_LANGUAGE](sample["source_text"]).to(dtype=torch.int64)
        tgt_sample = text_transform[TGT_LANGUAGE](sample["target_text"]).to(dtype=torch.int64)

        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    src_batch = truncate(src_batch)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = truncate(tgt_batch)

    return src_batch, tgt_batch

def get_dataloader(source_file, target_file, vocab_transform, batch_size, mode="train", num_workers=4):
    """
    Creates a DataLoader for the translation dataset.

    Args:
        source_file (str): Path to the source (Vietnamese) text file.
        target_file (str): Path to the target (English) text file.
        batch_size (int, optional): Batch size (default=32).
        shuffle (bool, optional): Whether to shuffle the dataset (default=True).
        num_workers (int, optional): Number of workers for data loading (default=4).

    Returns:
        DataLoader: A PyTorch DataLoader for batching the dataset.
    """
    dataset = TranslationDataset(source_file, target_file)
    text_transform = get_text_transforms(vocab_transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=num_workers,
                      collate_fn=lambda batch: collate_fn(batch, text_transform))


