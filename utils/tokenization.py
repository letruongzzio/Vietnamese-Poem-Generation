import os
import sys
import re
import concurrent.futures
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import *


def custom_tokenize(text):
    """
    Custom tokenizer:
    1) Separate special tokens while keeping them intact.
    2) Use underthesea word_tokenize for the remaining parts.
    3) Combine all into the final list of tokens.
    """
    # Regex to separate and "capture" special tokens
    # (?:...) is a non-capturing group, while (...) is a capturing group
    # Here we want to capture (S|EOL|EOS...) => pattern = (<sos>|<eol>|<eos>|<unk>|<pad>)
    pattern = r'(' + '|'.join(map(re.escape, SPECIAL_SYMBOLS)) + r')'
    
    # re.split(..., text) with capturing group will give us a list 
    # alternating between normal text segments and special tokens
    parts = re.split(pattern, text)
    
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            # Skip empty segments
            continue
        
        # If this part is one of the special tokens
        if part in SPECIAL_SYMBOLS:
            tokens.append(part)
        else:
            # Otherwise, tokenize using Underthesea
            # Note: You can choose to word_tokenize, or split by characters...
            sub_tokens = word_tokenize(part)
            tokens.extend(sub_tokens)
    
    return tokens


def yield_tokens(df):
    """
    Yields tokenized words from the dataset for vocabulary building.
    """
    for _, row in df.iterrows():
        for token in custom_tokenize(row['content']):  # Yield individual tokens
            yield token

def build_vocabulary(df, min_freq=2, num_workers=4):
    """
    Builds a vocabulary for a single language.

    Args:
        df (Dataset): The dataset object.
        min_freq (int): Minimum frequency of words to be included in vocab.

    Returns:
        torchtext.vocab.Vocab: Vocabulary object for the language.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        vocab = build_vocab_from_iterator(
            executor.map(yield_tokens, [df]),
            min_freq=min_freq,
            specials=SPECIAL_SYMBOLS,
            special_first=True
        )
    vocab.set_default_index(UNK_IDX)
    return vocab
