import os
import sys
import requests
import zipfile
PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_DIR)
from constants import TRANSLATION_DATA_DIR

def download_trans_dataset(
        url: str, 
        target_dir: str = TRANSLATION_DATA_DIR
        ) -> str:
    """
    Download a machine-translation dataset from a given URL and extract it to the target directory.

    Args:
        url (str): Direct URL to the dataset (e.g., OPUS dataset link).
        target_dir (str): Path to the directory where the dataset should be saved.

    Returns:
        str: Path to the extracted dataset.
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.join(target_dir, url.split("/")[-1])
        # Download file
        response = requests.get(url, stream=True, timeout=10)
        # Check if the request was successful
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=5242880):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}")
            return ""
        # Extract file if it's a zip
        if filename.endswith(".zip"):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"Extracted to: {target_dir}")
        return target_dir
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return ""
    
# if __name__ == "__main__":
#     url = "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-vi.txt.zip"
#     download_trans_dataset(url)
