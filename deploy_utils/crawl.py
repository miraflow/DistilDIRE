import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.request import urlretrieve
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Function to fetch and parse the webpage
def fetch_images(url, cls_name):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=1400,1500")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("enable-automation")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    options.add_argument(f"user-agent={user_agent}")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(5)
    img_elements = driver.find_elements(By.CLASS_NAME, cls_name)
    urls = []
    for img in img_elements:
        src_url = img.get_attribute('src')
        urls.append(src_url)
    return urls


# Function to download images
def download_images(img_urls, download_folder='downloaded_images', desc=""):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for img_url in tqdm(img_urls, desc=desc):
        fname = img_url.split('/')[-1]
        fname = fname.split('?')[0]
        filename = os.path.join(download_folder, fname)
        urlretrieve(img_url, filename)
        print(f"Downloaded: {filename}")

# Main function to coordinate the image crawling
def main():
    page_n=0
    while True:
        page_n += 1
        url = f'https://www.freepik.com/search?ai=only&format=search&last_filter=page&last_value={page_n}&page={page_n}&query=people%2C+political'
        cls_name = "_1286nb17"
        download_path = "/truemedia-eval/crawled-fakes/images/fakes"
        try:
            img_urls = fetch_images(url, cls_name)
            if len(img_urls) == 0:
                break
            download_images(img_urls, download_path, desc=f"Page {page_n}")
            
        except:
            break

if __name__ == "__main__":
    main()
