import os
import sys
import requests
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

# Ensure that DATA_DIR is defined in the constants module (or replace with an appropriate path)
PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import DATA_DIR

def crawl_data(WEBDRIVER_DELAY_TIME_INT: int = 10, NUM_PAGES: int = 10) -> None:
    """
    Crawl data from "thivien.net" and save the results to a CSV file.
    
    Parameters:
        WEBDRIVER_DELAY_TIME_INT: Maximum wait time for WebDriver (seconds)
        NUM_PAGES: Number of pages to crawl
    """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")      # Run in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")    # Bypass OS security restrictions
    chrome_options.headless = True
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(WEBDRIVER_DELAY_TIME_INT)
    wait = WebDriverWait(driver, WEBDRIVER_DELAY_TIME_INT)

    poem_dataset = []
    poem_id = 0
    # Script to remove unwanted tags
    delete_script = 'arguments[0].parentNode.removeChild(arguments[0]);'

    for page_idx in tqdm(range(1, NUM_PAGES + 1)):
        main_url = f'https://www.thivien.net/searchpoem.php?PoemType=9&ViewType=1&Country=2&Page={page_idx}'
        driver.get(main_url)

        # Use relative xpath to get all poem blocks
        content_tags_xpath = '//*[@class="page-content container"]/div[2]/div/div[@class="list-item"]'
        content_tags = driver.find_elements(By.XPATH, content_tags_xpath)
        print(f"Page {page_idx}: found {len(content_tags)} poems.")

        # Extract necessary information (title, url) from the list page
        poem_infos = []
        for content_tag in content_tags:
            try:
                # Use relative xpath ('.//') to find elements inside content_tag
                poem_title_el = content_tag.find_element(By.XPATH, './/h4/a')
                poem_title = poem_title_el.text
                poem_url = poem_title_el.get_attribute('href')
                poem_infos.append({
                    'title': poem_title,
                    'url': poem_url
                })
            except Exception as e:
                print("Error getting title/url:", e)
                continue

        # Iterate through the list of poems collected
        for info in poem_infos:
            try:
                driver.get(info['url'])
                # Wait until the element with class 'poem-content' appears
                poem_content_tag = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'poem-content')))
                
                # Remove unwanted tags (e.g., <i> and <b>)
                try:
                    poem_content_i_tag = poem_content_tag.find_element(By.TAG_NAME, 'i')
                    driver.execute_script(delete_script, poem_content_i_tag)
                except NoSuchElementException:
                    pass

                try:
                    poem_content_b_tag = poem_content_tag.find_element(By.TAG_NAME, 'b')
                    driver.execute_script(delete_script, poem_content_b_tag)
                except NoSuchElementException:
                    pass

                poem_content = poem_content_tag.text

                # Determine genre: if the number of words per line is inconsistent, it's 'Free verse', otherwise take the number of words of the longest line
                lines = poem_content.split('\n')
                line_lengths = [len(line.split()) for line in lines]
                max_words = max(line_lengths, default=0)
                genre_mark = 'Free verse' if any(line_lengths[i] != line_lengths[i-1] for i in range(1, len(line_lengths))) else f'{max_words} words'

                poem_id += 1
                poem_info = {
                    'id': poem_id,
                    'title': info['title'],
                    'url': info['url'],
                    'content': poem_content,
                    'genre': genre_mark
                }
                poem_dataset.append(poem_info)

                # Go back to the list page to process the next poem
                driver.back()
            except Exception as e:
                print("Error processing poem:", info['url'], e)
                continue

    driver.quit()

    # Save data to CSV file
    df = pd.DataFrame(poem_dataset)
    output_path = os.path.join(DATA_DIR, 'thivien_poem.csv')
    df.to_csv(output_path, index=True, encoding='utf-8-sig')
    print("Data crawled successfully! File saved at:", output_path)

