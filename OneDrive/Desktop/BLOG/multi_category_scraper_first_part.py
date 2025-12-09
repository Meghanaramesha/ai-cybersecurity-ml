from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import markdownify
import os
from urllib.parse import urljoin
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# ---------------- CONFIG ----------------
BASE_FOLDER = r"C:\Users\HP\Desktop\Blog"
CATEGORY = "UA_Growth_Agencies"
BLOG_URL = "https://www.gamigion.com/author/kopelovich/ "
MAX_THREADS = 5  # Parallel scraping threads
# ----------------------------------------

def create_category_folder():
    folder_path = os.path.join(BASE_FOLDER, CATEGORY)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_all_article_links(blog_url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(blog_url)
    time.sleep(3)

    # Scroll repeatedly to load all articles
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count += 1
        if new_height == last_height or scroll_count > 20:  # stop after 20 scrolls max
            break
        last_height = new_height

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        if href.startswith("/journal/") or href.startswith("/inside-"):
            full_url = urljoin(blog_url, href)
            links.add(full_url)
    return list(links)

def get_article_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find("div", class_="article-content") or soup.find("article") or soup.body
        if not content:
            return None
        for img in content.find_all("img"):
            if img.has_attr("src"):
                img['src'] = urljoin(url, img['src'])
        title = soup.title.string.strip() if soup.title else "Untitled Article"
        md_content = markdownify.markdownify(str(content), heading_style="ATX")
        return title, md_content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_article(title, md_content, folder):
    safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
    path = os.path.join(folder, f"{safe_title}.md")
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{md_content}")

def scrape_article(url, folder, failed_urls):
    result = get_article_content(url)
    if result:
        title, md = result
        save_article(title, md, folder)
        print(f"✅ Saved: {title}")
    else:
        failed_urls.append(url)
        print(f"❌ Failed: {url}")

def main():
    folder = create_category_folder()
    failed_urls = []

    print(f"🔍 Fetching all articles from {BLOG_URL}")
    article_links = get_all_article_links(BLOG_URL)
    print(f"📌 Found {len(article_links)} articles")

    # Multithreaded scraping
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for link in article_links:
            executor.submit(scrape_article, link, folder, failed_urls)

    if failed_urls:
        with open(os.path.join(folder, "failed_urls.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(failed_urls))
        print(f"⚠️ Failed URLs logged: {len(failed_urls)}")

if __name__ == "__main__":
    main()
