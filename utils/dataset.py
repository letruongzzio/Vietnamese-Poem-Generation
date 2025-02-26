import os
import sys
import concurrent.futures
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import PAD_IDX

def pad_and_truncate(input_ids, max_seq_len):
    """
    Pads or truncates a list of token indices to a fixed length.

    Args:
        input_ids (list): List of token indices.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list: Padded or truncated list of token indices.
    """
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
    else:
        input_ids += [PAD_IDX] * (max_seq_len - len(input_ids))
    return input_ids

def vectorizer(text, max_seq_len, tokenizer, vocab):
    """
    Converts input text to a sequence of token indices using the provided tokenizer and vocabulary,
    and pads or truncates the sequence to the specified maximum length.

    Args:
        text (str): Input text.
        max_seq_len (int): Maximum sequence length.
        tokenizer (callable): Function to tokenize the text.
        vocab (dict): Dictionary mapping tokens to indices.

    Returns:
        list: List of token indices.
    """
    tokens = tokenizer(text)
    input_ids = [vocab[token] for token in tokens]
    input_ids = pad_and_truncate(input_ids, max_seq_len)
    return input_ids

def create_padding_mask(input_ids, pad_token_id=PAD_IDX):
    """
    Creates a padding mask for the sequence of token indices.
    Returns 0 for pad tokens and 1 for non-pad tokens.

    Args:
        input_ids (list): List of token indices.
        pad_token_id (int): The index used for padding (default: PAD_IDX).

    Returns:
        list: A list representing the padding mask.
    """
    return [0 if token_id == pad_token_id else 1 for token_id in input_ids]

def split_content(content):
    """
    Splits the full content of a poem into samples.
    The content is first split by two consecutive newline characters,
    and each part is then split by a single newline.

    Args:
        content (str): Full text content of a poem.

    Returns:
        list: A list of samples, where each sample is a list of lines.
    """
    samples = []
    # Split by two newlines
    poem_parts = content.split('\n\n')
    for poem_part in poem_parts:
        poem_in_lines = poem_part.split('\n')
        samples.append(poem_in_lines)
    return samples

def prepare_sample(sample, tokenizer, vocab, max_seq_len):
    """
    Prepares input sequences, target sequences, and padding masks from a single sample.

    The sample (a list of lines) is converted into a single string by concatenating the lines 
    with a special end-of-line token ("<eol>"). A start-of-sequence ("<sos>") token is added at 
    the beginning and an end-of-sequence ("<eos>") token at the end.
    
    Then, for each token position, an input sequence is generated (tokens up to the current position)
    and a target sequence is generated (tokens shifted by one). Each sequence is vectorized (converted to 
    token indices) and padded/truncated to the fixed length.

    Args:
        sample (list): List of lines from the poem.
        tokenizer (callable): Function to tokenize text.
        vocab (dict): Dictionary mapping tokens to indices.
        max_seq_len (int): Maximum sequence length for each sample.

    Returns:
        tuple: Three lists: input sequences, target sequences, and padding masks.
    """
    input_seqs = []
    target_seqs = []
    padding_masks = []
    # Construct the input text with special tokens
    input_text = '<sos> ' + ' <eol> '.join(sample) + ' <eol> <eos>'
    tokens = tokenizer(input_text)
    for idx in range(1, len(tokens)):
        # Create input sequence (tokens from start to idx-1) and target sequence (tokens from 1 to idx)
        input_seq_text = ' '.join(tokens[:idx])
        target_seq_text = ' '.join(tokens[1:idx+1])
        inp = vectorizer(input_seq_text, max_seq_len, tokenizer, vocab)
        tgt = vectorizer(target_seq_text, max_seq_len, tokenizer, vocab)
        mask = create_padding_mask(inp)
        input_seqs.append(inp)
        target_seqs.append(tgt)
        padding_masks.append(mask)
    return input_seqs, target_seqs, padding_masks

def process_row_helper(row, tokenizer, vocab, max_seq_len):
    """
    Processes a single DataFrame row to extract and prepare samples.
    
    It splits the 'content' field into multiple samples and prepares input sequences,
    target sequences, and corresponding padding masks for each sample.

    Args:
        row (pd.Series or dict): A row from the DataFrame containing at least a 'content' field.
        tokenizer (callable): Function to tokenize text.
        vocab (dict): Dictionary mapping tokens to indices.
        max_seq_len (int): Maximum sequence length.

    Returns:
        tuple: Three lists of samples: input sequences, target sequences, and padding masks.
    """
    all_input_seqs = []
    all_target_seqs = []
    all_padding_masks = []
    content = row['content']
    samples = split_content(content)
    for sample in samples:
        inp, tgt, mask = prepare_sample(sample, tokenizer, vocab, max_seq_len)
        all_input_seqs.extend(inp)
        all_target_seqs.extend(tgt)
        all_padding_masks.extend(mask)
    return all_input_seqs, all_target_seqs, all_padding_masks

class PoemDataset(Dataset):
    """
    A custom Dataset class for processing poem data.

    This dataset takes a DataFrame with a 'content' column and uses a tokenizer and a vocabulary
    to generate vectorized input sequences, target sequences, and corresponding padding masks.
    Multi-threading is used to process each row concurrently for faster sample creation.

    Attributes:
        tokenizer (callable): Function to tokenize text.
        vocab (dict): Dictionary mapping tokens to their indices.
        max_seq_len (int): Maximum sequence length for each sample.
        input_seqs (torch.Tensor): Tensor containing all input sequences.
        target_seqs (torch.Tensor): Tensor containing all target sequences.
        padding_masks (torch.Tensor): Tensor containing all padding masks.
    """
    def __init__(self, df, tokenizer, vocab, max_seq_len=25):
        """
        Initializes the PoemDataset.

        Args:
            df (pd.DataFrame): DataFrame containing the poem data with a 'content' column.
            tokenizer (callable): Tokenizer function.
            vocab (dict): Vocabulary dictionary mapping tokens to indices.
            max_seq_len (int, optional): Maximum sequence length for each sample. Defaults to 25.
        """
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.input_seqs, self.target_seqs, self.padding_masks = self.create_samples(df)

    def create_samples(self, df):
        """
        Creates samples (input sequences, target sequences, and padding masks) from the DataFrame.

        Utilizes multi-threading (ThreadPoolExecutor) to process each row in parallel.

        Args:
            df (pd.DataFrame): DataFrame with a 'content' column.

        Returns:
            tuple: Three torch.Tensors (input_seqs, target_seqs, padding_masks).
        """
        all_input_seqs = []
        all_target_seqs = []
        all_padding_masks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_row_helper, row, self.tokenizer, self.vocab, self.max_seq_len)
                       for _, row in df.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                inp, tgt, mask = future.result()
                all_input_seqs.extend(inp)
                all_target_seqs.extend(tgt)
                all_padding_masks.extend(mask)
        input_seqs = torch.tensor(all_input_seqs, dtype=torch.long)
        target_seqs = torch.tensor(all_target_seqs, dtype=torch.long)
        padding_masks = torch.tensor(all_padding_masks, dtype=torch.float)
        return input_seqs, target_seqs, padding_masks

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.input_seqs)

    def __getitem__(self, idx):
        """
        Retrieves the sample (input sequence, target sequence, padding mask) at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input sequence, target sequence, padding mask) for the given sample.
        """
        return self.input_seqs[idx], self.target_seqs[idx], self.padding_masks[idx]
