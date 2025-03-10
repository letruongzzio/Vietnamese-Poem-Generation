import os

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
STORAGE_DIR = os.path.join(PROJECT_ROOT, 'storage')
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX, EOL_IDX = 0, 1, 2, 3, 4
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>', '<eol>']