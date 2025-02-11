import os
import sys
import concurrent.futures
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from translation_dataset import TranslationDataset

PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_DIR)
from constants import SRC_LANGUAGE, TGT_LANGUAGE, UNK_IDX, SPECIAL_SYMBOLS


def get_token_transforms():
    """
    Returns a dictionary containing tokenizers for both source (English) 
    and target (Vietnamese) languages.
    """
    def vi_tokenizer(sentence):
        """
        Tokenizes Vietnamese text using the underthesea library.

        Args:
            sentence (str): Input sentence.

        Returns:
            List of tokens.
        """
        return word_tokenize(sentence)
    
    return {
        SRC_LANGUAGE: get_tokenizer('basic_english'),
        TGT_LANGUAGE: get_tokenizer(vi_tokenizer)
    }

def yield_tokens(dataset, lang, token_transform):
    """
    Yields tokenized sentences from the dataset for vocabulary building.

    Args:
        dataset (Dataset): The dataset object.
        lang (str): Language key ('en' or 'vi').
        token_transform (dict): Tokenizer dictionary.

    Yields:
        List of tokens from the dataset.
    """
    lang_key = "source_text" if lang == SRC_LANGUAGE else "target_text"
    for sample in dataset:
        yield token_transform[lang](sample[lang_key])

def build_vocab_for_lang(dataset, lang, token_transform, min_freq):
    """
    Builds a vocabulary for a single language.

    Args:
        dataset (Dataset): The dataset object.
        lang (str): Language key ('en' or 'vi').
        token_transform (dict): Tokenizer dictionary.
        min_freq (int): Minimum frequency of words to be included in vocab.

    Returns:
        torchtext.vocab.Vocab: Vocabulary object for the language.
    """
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset, lang, token_transform),
        min_freq=min_freq,
        specials=SPECIAL_SYMBOLS,
        special_first=True
    )
    vocab.set_default_index(UNK_IDX)
    return vocab

def build_vocabulary(source_file, target_file, min_freq=1, num_workers=4):
    """
    Builds vocabularies for both source and target languages in parallel.

    Args:
        source_file (str): Path to the source (English) text file.
        target_file (str): Path to the target (Vietnamese) text file.
        min_freq (int, optional): Minimum frequency of words to be included in vocab (default=1).

    Returns:
        dict: Dictionary containing vocab objects for both languages.
    """
    dataset = TranslationDataset(source_file, target_file)
    token_transform = get_token_transforms()
    vocab_transform = {}

    # Use multi-threading to build vocab for both languages simultaneously
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        future_en = executor.submit(build_vocab_for_lang, dataset, SRC_LANGUAGE, token_transform, min_freq)
        future_vi = executor.submit(build_vocab_for_lang, dataset, TGT_LANGUAGE, token_transform, min_freq)

        vocab_transform[SRC_LANGUAGE] = future_en.result()
        vocab_transform[TGT_LANGUAGE] = future_vi.result()

    return vocab_transform
