import os
import sys
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from constants import *

if __name__ == "__main__":
    data_downloading = load_dataset("Libosa2707/vietnamese-poem")
    data_downloading['train'].to_csv(os.path.join(DATA_DIR, 'vietnamese_poem.csv'), index=False)
    print("Data downloaded successfully!")
    df_1 = pd.read_csv(os.path.join(DATA_DIR, 'vietnamese_poem.csv'))