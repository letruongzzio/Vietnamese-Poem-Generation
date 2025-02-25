import os
import sys
import pandas as pd
from datasets import load_dataset
from data_crawling import crawl_data

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import DATA_DIR

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if __name__ == "__main__":
    # 1. Download available dataset from Hugging Face
    # data_downloading = load_dataset("Libosa2707/vietnamese-poem")
    # data_downloading['train'].to_csv(os.path.join(DATA_DIR, 'vietnamese_poem.csv'), index=False)
    # print("Data downloaded successfully!")
    df_1 = pd.read_csv(os.path.join(DATA_DIR, 'vietnamese_poem.csv'))

    # 2. Crawl data from "thivien.net"
    # crawl_data(WEBDRIVER_DELAY_TIME_INT=30, NUM_PAGES=10)
    df_2 = pd.read_csv(os.path.join(DATA_DIR, 'thivien_poem.csv'))

    # 3. Merge two datasets
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = df.drop_duplicates(subset=['title', 'content']).set_index('id').drop(columns=['Unnamed: 0'])
    df.to_csv(os.path.join(DATA_DIR, 'poem_dataset.csv'), index=False)
    print("Data setup successfully!")
    