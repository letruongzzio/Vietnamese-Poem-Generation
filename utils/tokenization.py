import os
import sys
import concurrent.futures
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import *


def yield_tokens(df):
    """
    Yields tokenized words from the dataset for vocabulary building.
    """
    for _, row in df.iterrows():
        for token in word_tokenize(row['content']):  # Yield individual tokens
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


