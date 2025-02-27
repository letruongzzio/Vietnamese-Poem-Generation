import os
import sys
import torch
from torch.utils.data import Dataset
import concurrent.futures

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import PAD_IDX

def pad_and_truncate(input_ids, max_seq_len):
    """
    Pad or truncate a list of token indices to a fixed length.

    Args:
        input_ids (list): List of token indices.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list: List of token indices padded or truncated to max_seq_len.
    """
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
    else:
        input_ids += [PAD_IDX] * (max_seq_len - len(input_ids))
    return input_ids

def vectorizer(text, max_seq_len, tokenizer, vocab):
    """
    Convert input text to a sequence of token indices using the provided tokenizer and vocabulary,
    then pad or truncate the sequence to a fixed length.

    Args:
        text (str): Input text.
        max_seq_len (int): Maximum sequence length.
        tokenizer (callable): A function that tokenizes text.
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
    Create a padding mask for the given sequence of token indices.
    Returns 0 for pad tokens and 1 for non-pad tokens.

    Args:
        input_ids (list): List of token indices.
        pad_token_id (int): Padding token index (default: PAD_IDX).

    Returns:
        list: Padding mask (list of 0s and 1s).
    """
    return [0 if token_id == pad_token_id else 1 for token_id in input_ids]

def split_content(content):
    """
    Split the poem content into samples.
    The content is split by two consecutive newline characters into parts,
    then each part is further split by a single newline.

    Args:
        content (str): Full text content of a poem.

    Returns:
        list: List of samples, each sample is a list of lines.
    """
    samples = []
    poem_parts = content.split('\n\n')
    for poem_part in poem_parts:
        poem_in_lines = poem_part.split('\n')
        samples.append(poem_in_lines)
    return samples

def prepare_sample(sample, tokenizer, vocab, max_seq_len):
    """
    Prepare input sequences, target sequences, and padding masks from a single sample.

    The sample (a list of lines) is concatenated into a string with special tokens:
    a start-of-sequence ("<sos>"), an end-of-line ("<eol>") between lines, and an end-of-sequence ("<eos>").
    Then for each token position, an input sequence (tokens up to the current position) and a target sequence 
    (tokens shifted by one) are generated. These sequences are vectorized and padded/truncated.

    Args:
        sample (list): List of lines from a poem.
        tokenizer (callable): Tokenizer function.
        vocab (dict): Vocabulary dictionary mapping tokens to indices.
        max_seq_len (int): Maximum sequence length.

    Returns:
        tuple: Three lists: input sequences, target sequences, and padding masks.
    """
    input_seqs = []
    target_seqs = []
    padding_masks = []
    input_text = '<sos> ' + ' <eol> '.join(sample) + ' <eol>' + ' <eos>'
    tokens = tokenizer(input_text)
    for idx in range(1, len(tokens)):
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
    Process a single DataFrame row to extract and prepare samples.
    
    This function splits the 'content' field into multiple samples and prepares input sequences,
    target sequences, and padding masks for each sample. It also ensures that, if using Underthesea's
    word_tokenize, the model is loaded by calling it with a dummy text.

    Args:
        row (pd.Series or dict): A row from the DataFrame with a 'content' field.
        tokenizer (callable): Tokenizer function.
        vocab (dict): Vocabulary mapping tokens to indices.
        max_seq_len (int): Maximum sequence length.

    Returns:
        tuple: Three lists (input sequences, target sequences, padding masks) for the row.
    """
    # If using Underthesea's word_tokenize, force model initialization in this process.
    if hasattr(tokenizer, '__name__') and tokenizer.__name__ == 'word_tokenize':
        import underthesea
        _ = underthesea.word_tokenize("Xin chào")
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

    This dataset converts the 'content' column of a DataFrame into vectorized input sequences,
    target sequences, and corresponding padding masks using a given tokenizer and vocabulary.
    Multi-processing (ProcessPoolExecutor) is used to speed up the processing of each row.

    Attributes:
        tokenizer (callable): Function to tokenize text.
        vocab (dict): Vocabulary dictionary mapping tokens to indices.
        max_seq_len (int): Maximum sequence length for each sample.
        input_seqs (torch.Tensor): Tensor containing all input sequences.
        target_seqs (torch.Tensor): Tensor containing all target sequences.
        padding_masks (torch.Tensor): Tensor containing all padding masks.
    """
    def __init__(self, df, tokenizer, vocab, max_seq_len=25):
        """
        Initialize the PoemDataset.

        Args:
            df (pd.DataFrame): DataFrame containing poem data with a 'content' column.
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
        Create samples (input sequences, target sequences, and padding masks) from the DataFrame.

        Uses ProcessPoolExecutor to process each row in parallel.

        Args:
            df (pd.DataFrame): DataFrame with a 'content' column.

        Returns:
            tuple: Three torch.Tensors (input_seqs, target_seqs, padding_masks).
        """
        all_input_seqs = []
        all_target_seqs = []
        all_padding_masks = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
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
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.input_seqs)

    def __getitem__(self, idx):
        """
        Retrieve the sample (input sequence, target sequence, padding mask) at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input sequence, target sequence, padding mask) for the sample.
        """
        return self.input_seqs[idx], self.target_seqs[idx], self.padding_masks[idx]
