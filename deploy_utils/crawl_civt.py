import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib.request
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Function to fetch and parse the webpage
def fetch_images(url):
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
    driver.set_window_size(1400, 1500)
    driver.get(url)
    time.sleep(5)
    last_height = driver.execute_script("return document.scrollingElement.scrollHeight")
    urls = []
    while True :
        # img_elements = driver.find_elements(By.CLASS_NAME, cls_name)
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        for img in img_elements:
            urls.append(img.get_attribute('src') or img.get_attribute('data-src'))
        driver.execute_script(f"document.scrollingElement.scrollTo(0, document.scrollingElement.scrollHeight);")
        time.sleep(5)
        new_height = driver.execute_script("return document.scrollingElement.scrollHeight")
        print(f"last_height: {last_height}, new_height: {new_height}")
        if new_height == last_height:
            break
            
        last_height = new_height
    
    print(f"Total images: {len(urls)}")
    
    return urls


# Function to download images
def download_images(img_urls, download_folder='downloaded_images', desc=""):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for img_url in tqdm(img_urls, desc=desc):
        fname = img_url.split('/')[-1]
        fname = fname.split('?')[0]
        filename = os.path.join(download_folder, fname)
        if os.path.exists(filename):
            continue
        req = urllib.request.Request(img_url, headers=headers)
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)



def crawl(query):
    url = f'https://openart.ai/search/{query}?method=similarity'
    download_path = "/truemedia-eval/crawled-fakes/images/fakes"
    try:
        img_urls = fetch_images(url)
        if len(img_urls) == 0:
            return
        download_images(img_urls, download_path, desc=f"query: {query}")
    except:
        return

# Main function to coordinate the image crawling
def main():
    # queries = ['xi', 'trump', 'biden', 'putin', 'zelensky', 'gaza', 'israel', 'taylor', 'palesti', 'nsfw', 'instagram']
    queries = ['xi', 'trump', 'biden', 'putin', 'zelensky', 'gaza', 'israel', 'taylor', 'palesti', 'nsfw', 'instagram',
    "Hamas", "Hezbollah", "Iran",
    "North Korea", "Kim Jong-un", "Uyghurs", "Hong Kong", "Taiwan", "Tibet", "Kashmir", "Syria", "Assad", "ISIS",
    "Taliban", "Afghanistan", "Saudi", "Yemen", "Qatar", "Brexit", "Catalonia", "Black lives matter", "Antifa", "Proud Boys",
    "Abortion", "Guns", "Second Amendment", "Police", "Execution", "Drugs", "Opioids", "Climate", "Paris",
    "Green", "Keystone", "Fracking", "GMOs", "Net", "WikiLeaks", "Assange", "Snowden", "NSA", "Patriot",
    "COVID", "Vaccines", "Masks", "5G", "QAnon", "Fraud", "Capitol", "Impeachment", "Hunter", "Epstein",
    "Trafficking", "MeToo", "Weinstein", "Cancel", "Race", "Affirmative", "Transgender", "Marriage",
    "Religion", "Nationalism", "Neo-Nazism", "Confederate", "Antisemitism", "Islamophobia", "Wall", "ICE",
    "DACA", "Dreamers", "Separation", "Sanctuary", "Refugees", "Travel", "Benghazi", "Emails", "Russia",
    "Mueller", "Ukraine", "Mar-a-Lago", "Taxes", "Emoluments", "Gerrymandering", "Voter", "Mail", "Citizens",
    "PACs", "Breonna", "Floyd", "Rittenhouse", "Insurrection", "Ivanka", "Kushner", "Bannon", "Flynn", "Stone"
]

    for query in queries:
        crawl(query)
    

if __name__ == "__main__":
    main()
