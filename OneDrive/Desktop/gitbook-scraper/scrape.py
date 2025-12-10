"""
scrape.py
Full-featured GitBook scraper for multiple blogs:
- Crawls all discovered pages recursively (articles, journals, blog posts).
- Handles pagination automatically.
- Concurrent downloads with retries + backoff.
- Saves per-article content as Markdown in Agency/<Blog>/.
- Maintains JSON cache to avoid re-scraping unchanged pages.
- Generates README.md files for blogs and Agency index.
"""

import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional, Tuple, List, Dict, Any, Set

import requests
from bs4 import BeautifulSoup
import markdownify
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ----------------
BLOGS = {
    "Lanceric": ("https://lancaric.me/blog/", 28),  # (base_url, total_pages)
    "RamizTrtovac": ("https://ramiztrtovac.com/blog/", None),
    "Kopelovich": ("https://www.gamigion.com/journal/", None),
    "RZain": ("https://rzain.blog/games/", None),
}

OUTPUT_ROOT = Path("GitBook_Content")
AGENCY_DIR = OUTPUT_ROOT / "Agency"
CACHE_FILE = OUTPUT_ROOT / ".scrape_cache.json"
LOG_FILE = OUTPUT_ROOT / "scrape.log"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GitBookScraper/1.0; +https://example.com)"
}

MAX_WORKERS = 10
REQUEST_TIMEOUT = 12
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5

# Allowed filename characters
FNAME_SAFE = re.compile(r'[^A-Za-z0-9\-_ ]')

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ---------------- UTIL ----------------
def ensure_folder(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_filename(title: str) -> str:
    name = FNAME_SAFE.sub("", title).strip()
    return name[:200] or "untitled"

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to read cache, starting fresh.")
    return {"pages": {}}

def save_cache(cache: Dict[str, Any]):
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

# ---------------- HTTP ----------------
session = requests.Session()
session.headers.update(HEADERS)

def get_html(url: str) -> Optional[str]:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.text
            else:
                logging.warning("Non-200 %s for %s", r.status_code, url)
                return None
        except requests.RequestException as e:
            logging.warning("Request error %s (attempt %d) for %s", e, attempt, url)
            time.sleep(delay)
            delay *= RETRY_BACKOFF
    logging.error("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None

# ---------------- LINK EXTRACTION ----------------
def extract_links_from_listing(html: str, base_url: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()

    # Article tags
    for art in soup.find_all("article"):
        a = art.find("a", href=True)
        if a:
            links.add(urljoin(base_url, a["href"]))

    # Anchor style links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if urlparse(full).netloc == urlparse(base_url).netloc:
            p = urlparse(full).path.lower()
            if any(x in p for x in ("/20", "/post", "/article", "/blog", "/author")) or len(p.split("/")) > 2:
                links.add(full)

    # Canonical link
    link_tag = soup.find("link", {"rel": "canonical"})
    if link_tag and link_tag.get("href"):
        links.add(urljoin(base_url, link_tag["href"]))

    return {u.split("#")[0].rstrip("/") for u in links}

# ---------------- CONTENT EXTRACTION ----------------
def extract_full_content(html: str) -> Optional[Tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("meta", {"property": "og:title"}) or soup.find("title") or soup.find("h1")
    if getattr(title_tag, "get", None):
        title_text = title_tag.get("content", "").strip()
    else:
        title_text = title_tag.string.strip() if title_tag and title_tag.string else "Untitled"

    selectors = [
        ("article", None),
        ("div", {"class": "entry-content"}),
        ("div", {"class": "post-content"}),
        ("main", None),
        ("div", {"id": "content"}),
        ("section", {"class": "post"}),
    ]
    for tag, attrs in selectors:
        content = soup.find(tag, attrs) if attrs else soup.find(tag)
        if content and len(content.get_text(strip=True)) > 100:
            return title_text, str(content)

    body = soup.body
    if body and len(body.get_text(strip=True)) > 200:
        return title_text, str(body)

    return None

def markdownify_html(html_fragment: str) -> str:
    return markdownify.markdownify(html_fragment, heading_style="ATX")

def write_markdown(blog_folder: Path, title: str, markdown_text: str) -> str:
    fname = f"{clean_filename(title)}.md"
    path = blog_folder / fname
    content = f"# {title}\n\n{markdown_text}\n"
    path.write_text(content, encoding="utf-8")
    return fname

# ---------------- CORE SCRAPING ----------------
def process_single_url(url: str, blog_folder: Path, cache: Dict[str, Any]) -> Optional[str]:
    html = get_html(url)
    if not html:
        return None

    h = sha256_text(html)[:64]
    cached = cache["pages"].get(url)
    if cached and cached.get("hash") == h:
        cached["last_seen"] = int(time.time())
        logging.info("Unchanged: %s", url)
        return cached.get("path")

    extracted = extract_full_content(html)
    if not extracted:
        logging.info("No content found for %s", url)
        return None
    title, content_html = extracted

    md = markdownify_html(content_html)
    fname = write_markdown(blog_folder, title, md)

    cache["pages"][url] = {"hash": h, "path": fname, "last_seen": int(time.time())}
    logging.info("Scraped: %s -> %s", url, fname)
    return fname

def scrape_blog(blog_name: str, base_url: str, total_pages: Optional[int], cache: Dict[str, Any]) -> List[str]:
    logging.info("Scraping blog: %s (%s)", blog_name, base_url)
    blog_folder = AGENCY_DIR / blog_name
    ensure_folder(blog_folder)

    all_links: Set[str] = set()

    # Generate pagination URLs if total_pages provided
    pages_to_scrape = [base_url.rstrip("/")]
    if total_pages:
        for i in range(2, total_pages + 1):
            pages_to_scrape.append(f"{base_url.rstrip('/')}/page/{i}/")

    # Crawl each page and extract links
    for page_url in pages_to_scrape:
        html = get_html(page_url)
        if html:
            links = extract_links_from_listing(html, base_url)
            all_links |= links
            logging.info("Extracted %d links from %s", len(links), page_url)

    logging.info("Total %d article links to scrape for %s", len(all_links), blog_name)

    scraped_files = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_single_url, url, blog_folder, cache): url for url in all_links}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                scraped_files.append(res)

    create_blog_readme(blog_name, blog_folder, scraped_files)
    return scraped_files

# ---------------- README GENERATION ----------------
def create_main_readme():
    f = OUTPUT_ROOT / "README.md"
    content = (
        "# Influencer Blogs (GitBook)\n\n"
        "This repo is generated by `scrape.py` and structured for GitBook sync.\n\n"
        "## Agencies\n\n- [Agency](Agency/README.md)\n"
    )
    f.write_text(content, encoding="utf-8")

def create_agency_index(blog_names: Optional[List[str]] = None):
    blog_names = blog_names or list(BLOGS.keys())
    f = AGENCY_DIR / "README.md"
    lines = ["# Agency Blogs\n"]
    for b in blog_names:
        lines.append(f"- [{b}]({b}/README.md)")
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

def create_blog_readme(blog_name: str, blog_folder: Path, files: List[str]):
    f = blog_folder / "README.md"
    lines = [f"# {blog_name} Articles\n"]
    if not files:
        lines.append("No articles found.\n")
    else:
        for i, fn in enumerate(sorted(set(files)), start=1):
            lines.append(f"{i}. [{fn}]({fn})")
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------- MAIN ----------------
def main():
    ensure_folder(OUTPUT_ROOT)
    ensure_folder(AGENCY_DIR)

    cache = load_cache()
    all_blogs_files = {}

    for blog_name, (url, total_pages) in BLOGS.items():
        files = scrape_blog(blog_name, url, total_pages, cache)
        all_blogs_files[blog_name] = files

    create_main_readme()
    create_agency_index(list(all_blogs_files.keys()))
    save_cache(cache)
    logging.info("Done. Scraped %d blogs.", len(all_blogs_files))

if __name__ == "__main__":
    main()
