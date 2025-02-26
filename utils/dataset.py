import os
import sys
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import *

class PoemDataset(Dataset):
    """
    A custom Dataset class for handling poem data.

    Args:
        df (pd.DataFrame): DataFrame containing the poem data with a 'content' column.
        tokenizer (callable): A tokenizer function to tokenize the text.
        vocab (dict): A dictionary mapping tokens to their respective indices.
        max_seq_len (int, optional): Maximum sequence length for padding/truncating. Default is 128.

    Attributes:
        tokenizer (callable): Tokenizer function.
        vocab (dict): Vocabulary dictionary.
        max_seq_len (int): Maximum sequence length.
        input_seqs (torch.Tensor): Tensor of input sequences.
        target_seqs (torch.Tensor): Tensor of target sequences.
        padding_masks (torch.Tensor): Tensor of padding masks.

    Methods:
        pad_and_truncate(input_ids, max_seq_len):
            Pads or truncates the input_ids to the specified max_seq_len.
        
        vectorizer(text, max_seq_len):
            Converts text to a sequence of token indices and pads/truncates to max_seq_len.
        
        create_padding_mask(input_ids, pad_token_id=PAD_IDX):
            Creates a padding mask for the input_ids.
        
        split_content(content):
            Splits the content into samples based on double newline and single newline.
        
        prepare_sample(sample):
            Prepares input sequences, target sequences, and padding masks for a given sample.
        
        create_samples(df):
            Creates input sequences, target sequences, and padding masks from the DataFrame.
        
        __len__():
            Returns the number of samples in the dataset.
        
        __getitem__(idx):
            Returns the input sequence, target sequence, and padding mask at the specified index.
    """
    def __init__(self, df, tokenizer, vocab, max_seq_len=25):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.input_seqs, self.target_seqs, self.padding_masks = self.create_samples(df)

    def pad_and_truncate(self, input_ids, max_seq_len):
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
        else:
            input_ids += [PAD_IDX] * (max_seq_len - len(input_ids))
        return input_ids

    def vectorizer(self, text, max_seq_len):
        input_ids = [self.vocab[token] for token in self.tokenizer(text)]
        input_ids = self.pad_and_truncate(input_ids, max_seq_len)
        return input_ids

    def create_padding_mask(self, input_ids, pad_token_id=PAD_IDX):
        return [0 if token_id == pad_token_id else 1 for token_id in input_ids]

    def split_content(self, content):
        samples = []
        poem_parts = content.split('\n\n')
        for poem_part in poem_parts:
            poem_in_lines = poem_part.split('\n')
            samples.append(poem_in_lines)
        return samples

    def prepare_sample(self, sample):
        input_seqs = []
        target_seqs = []
        padding_masks = []

        input_text = '<sos> ' + ' <eol> '.join(sample) + ' <eol> <eos>'
        input_ids = self.tokenizer(input_text)
        for idx in range(1, len(input_ids)):
            input_seq = ' '.join(input_ids[:idx])
            target_seq = ' '.join(input_ids[1:idx+1])
            input_seq = self.vectorizer(input_seq, self.max_seq_len)
            target_seq = self.vectorizer(target_seq, self.max_seq_len)
            padding_mask = self.create_padding_mask(input_seq)

            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
            padding_masks.append(padding_mask)

        return input_seqs, target_seqs, padding_masks

    def create_samples(self, df):
        input_seqs = []
        target_words = []
        padding_masks = []

        for _, row in df.iterrows():
            content = row['content']
            samples = self.split_content(content)
            for sample in samples:
                sample_input_seqs, sample_target_words, sample_padding_masks = self.prepare_sample(sample)
                input_seqs += sample_input_seqs
                target_words += sample_target_words
                padding_masks += sample_padding_masks

        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        target_words = torch.tensor(target_words, dtype=torch.long)
        padding_masks = torch.tensor(padding_masks, dtype=torch.float)

        return input_seqs, target_words, padding_masks

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        input_seqs = self.input_seqs[idx]
        target_seqs = self.target_seqs[idx]
        padding_masks = self.padding_masks[idx]
        return input_seqs, target_seqs, padding_masks
    
    