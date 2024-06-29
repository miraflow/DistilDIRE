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
    last_hegiht = driver.execute_script("return document.body.scrollHeight")

    while True :
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_hegiht:
            break
        last_hegiht = new_height
        
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

def crawl(query):
    url = f'https://civitai.com/search/images?sortBy=images_v6%3Astats.reactionCountAllTime%3Adesc&query={query}'
    cls_name = "__mantine-ref-image mantine-deph6u"
    download_path = "/truemedia-eval/crawled-fakes/images/fakes"
    try:
        img_urls = fetch_images(url, cls_name)
        if len(img_urls) == 0:
            return
        download_images(img_urls, download_path, desc=f"query: {query}")
    except:
        return

# Main function to coordinate the image crawling
def main():
    queries = ['xi', 'trump', 'biden', 'putin', 'zelensky', 'gaza', 'israel', 'taylor', 'palesti', 'nsfw', 'instagram']
    for query in queries:
        crawl(query)
    

if __name__ == "__main__":
    main()
