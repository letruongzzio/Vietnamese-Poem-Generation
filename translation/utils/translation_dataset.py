import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file):
        """
        Dataset for Machine Translation (MT) tasks.

        Args:
            source_file (str): Path to the file containing source sentences (Vietnamese).
            target_file (str): Path to the file containing target sentences (English).
        """
        self.source_sentences, self.target_sentences, self.num_rows = self._load_files(source_file, target_file)

    def _load_files(self, source_file, target_file):
        """Reads sentences from both files, removing misaligned pairs."""
        num_rows = 0
        with open(source_file, 'r', encoding='utf-8') as src_f, open(target_file, 'r', encoding='utf-8') as tgt_f:
            source_lines = src_f.read().strip().split("\n")
            target_lines = tgt_f.read().strip().split("\n")
            num_rows += 1

        return source_lines, target_lines, num_rows

    def __len__(self):
        """Returns the total number of aligned sentence pairs."""
        return len(self.source_sentences)

    def __getitem__(self, idx):
        """Retrieves a data sample at index `idx`."""
        src_text = self.source_sentences[idx]
        tgt_text = self.target_sentences[idx]

        return {
            "source_text": src_text,
            "target_text": tgt_text
        }
