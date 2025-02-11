import os

PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
DATA_DIR = os.path.join(PROJECT_DIR, "data/")
TRANSLATION_DATA_DIR = os.path.join(DATA_DIR, "translation_data/")
TRANSLATION_TRAIN_DIR = os.path.join(TRANSLATION_DATA_DIR, "train/")
TRANSLATION_VAL_DIR = os.path.join(TRANSLATION_DATA_DIR, "val/")
TRANSLATION_TEST_DIR = os.path.join(TRANSLATION_DATA_DIR, "test/")
POEM_DATA_DIR = os.path.join(DATA_DIR, "poem_data/")
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
MAX_LEN = 830
BATCH_SIZE = [8, 16, 32, 64, 128]