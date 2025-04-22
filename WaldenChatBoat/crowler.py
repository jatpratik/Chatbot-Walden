import requests
from bs4 import BeautifulSoup
import time
import re
import os
import json
import urllib.parse
from collections import deque
import random


class WaldenUniversityScraper:
    def __init__(self, base_url="https://www.waldenu.edu/", output_folder="walden_data"):
        self.base_url = base_url
        self.visited_urls = set()
        self.queue = deque([base_url])
        self.output_folder = output_folder

        # Create a session with a realistic user agent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Initialize results file
        self.results_file = os.path.join(output_folder, "scraped_data.jsonl")
        with open(self.results_file, 'w') as f:
            pass  # Create empty file

        # Sitemap URLs to check
        self.sitemap_urls = [
            f"{self.base_url}sitemap.xml",
            f"{self.base_url}sitemap_index.xml",
            f"{self.base_url}sitemap/"
        ]

    def normalize_url(self, url, current_url):
        """Normalize URL to absolute form"""
        # Add domain if it's a relative URL
        if url.startswith('/'):
            return urllib.parse.urljoin(self.base_url, url)
        elif not url.startswith(('http://', 'https://')):
            return urllib.parse.urljoin(current_url, url)
        return url

    def should_process_url(self, url):
        """Determine if we should process this URL"""
        # Skip non-Walden URLs
        if not url.startswith(self.base_url):
            return False

        # Skip already visited URLs
        if url in self.visited_urls:
            return False

        # Skip URLs with fragments
        if '#' in url:
            clean_url = url.split('#')[0]
            if clean_url in self.visited_urls:
                return False

        # Skip file downloads and certain formats
        skip_extensions = ['.pdf', '.docx', '.xlsx', '.pptx', '.jpg', '.jpeg', '.png',
                           '.gif', '.css', '.js', '.xml', '.ico', '.svg']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False

        # Skip certain paths
        skip_patterns = ['/login', '/resource', '/search?', '/calendar', 'mailto:', 'tel:', '/cdn-cgi/']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False

        return True

    def extract_urls_from_sitemap(self, sitemap_url):
        """Extract URLs from a sitemap"""
        urls = []
        try:
            response = self.session.get(sitemap_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')

                # Check if it's a sitemap index
                sitemaps = soup.find_all('sitemap')
                if sitemaps:
                    for sitemap in sitemaps:
                        loc = sitemap.find('loc')
                        if loc:
                            child_sitemap_url = loc.text.strip()
                            child_urls = self.extract_urls_from_sitemap(child_sitemap_url)
                            urls.extend(child_urls)

                # Regular sitemap
                locations = soup.find_all('loc')
                for loc in locations:
                    url = loc.text.strip()
                    if url.startswith(self.base_url) and self.should_process_url(url):
                        urls.append(url)

        except Exception as e:
            print(f"Error processing sitemap {sitemap_url}: {e}")

        return urls

    def discover_urls_from_navigation(self):
        """Extract URLs from the site's navigation menus"""
        try:
            response = self.session.get(self.base_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Look for navigation elements
                nav_elements = soup.find_all(['nav', 'header', 'footer', 'div'])

                urls = []
                for nav in nav_elements:
                    for a_tag in nav.find_all('a', href=True):
                        href = a_tag.get('href', '').strip()
                        if href and not href.startswith(('javascript:', '#')):
                            absolute_url = self.normalize_url(href, self.base_url)
                            if self.should_process_url(absolute_url):
                                print("absolute_url",absolute_url)
                                urls.append(absolute_url)

                return urls

        except Exception as e:
            print(f"Error extracting navigation URLs: {e}")

        return []

    def extract_content(self, soup, url):
        """Extract meaningful content from the page"""
        # Initialize content dictionary
        content = {
            "title": "",
            "meta_description": "",
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "tables": [],  # Add tables field
            "url_path": urllib.parse.urlparse(url).path
        }

        # Get page title
        title_tag = soup.find('title')
        if title_tag:
            content["title"] = title_tag.get_text(strip=True)

        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content["meta_description"] = meta_desc.get('content').strip()

        # Get main content
        # Try to find the main content area using common selectors
        main_content = None
        for selector in ['main', 'article', '#content', '.content', '.main-content', 'section']:
            content_area = soup.select_one(selector)
            if content_area:
                main_content = content_area
                break

        # If still no content area found, use body
        if not main_content:
            main_content = soup.find('body')

        if main_content:
            # Extract headings
            for h_tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_text = h_tag.get_text(strip=True)
                if heading_text:
                    content["headings"].append({
                        "level": h_tag.name,
                        "text": heading_text
                    })

            # Extract paragraphs
            for p_tag in main_content.find_all('p'):
                p_text = p_tag.get_text(strip=True)
                if p_text and len(p_text) > 10:  # Skip very short paragraphs
                    content["paragraphs"].append(p_text)

            # Extract lists
            for list_tag in main_content.find_all(['ul', 'ol']):
                list_items = []
                for li in list_tag.find_all('li'):
                    li_text = li.get_text(strip=True)
                    if li_text:
                        list_items.append(li_text)
                if list_items:
                    content["lists"].append({
                        "type": list_tag.name,
                        "items": list_items
                    })

            # Extract tables
            for table in main_content.find_all('table'):
                table_data = {
                    "headers": [],
                    "rows": []
                }

                # Get table headers
                headers = table.find_all('th')
                if headers:
                    for header in headers:
                        header_text = header.get_text(strip=True)
                        if header_text:
                            table_data["headers"].append(header_text)

                # Get table rows
                for row in table.find_all('tr'):
                    row_data = []
                    cells = row.find_all(['td', 'th'])

                    # Skip rows with only header cells if we've already processed headers
                    if all(cell.name == 'th' for cell in cells) and table_data["headers"]:
                        continue

                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        row_data.append(cell_text)

                    if row_data:  # Only add non-empty rows
                        table_data["rows"].append(row_data)

                # Only add tables with actual data
                if table_data["headers"] or table_data["rows"]:
                    content["tables"].append(table_data)

        return content

    def save_content(self, url, content):
        """Save the extracted content to a JSONL file"""
        data = {
            "url": url,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')

    def extract_links(self, soup, current_url):
        """Extract all links from the page"""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()
            if href and not href.startswith(('javascript:', '#')):
                absolute_url = self.normalize_url(href, current_url)
                if self.should_process_url(absolute_url):
                    links.append(absolute_url)
        return links

    def crawl(self, max_pages=None):
        """Crawl the website"""
        # Try to get URLs from sitemap first
        sitemap_urls = []
        for sitemap_url in self.sitemap_urls:
            urls = self.extract_urls_from_sitemap(sitemap_url)
            if urls:
                sitemap_urls.extend(urls)
                print(f"Found {len(urls)} URLs from sitemap {sitemap_url}")

        # Get URLs from navigation
        nav_urls = self.discover_urls_from_navigation()
        print(f"Found {len(nav_urls)} URLs from navigation")

        # Add discovered URLs to the queue
        for url in nav_urls:
            if url not in self.visited_urls and url not in self.queue:
                self.queue.append(url)

        page_count = 0
        print(f"Starting to crawl {self.base_url}")
        print(f"Initial queue size: {len(self.queue)}")

        while self.queue and (max_pages is None or page_count < max_pages):
            current_url = self.queue.popleft()

            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)

            print(f"Crawling ({page_count + 1}): {current_url}")

            try:
                # Add a random delay between requests to be respectful
                time.sleep(random.uniform(1.0, 3.0))

                response = self.session.get(current_url, timeout=30)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract and save content
                    content = self.extract_content(soup, current_url)
                    self.save_content(current_url, content)

                    # Extract links and add to queue
                    links = self.extract_links(soup, current_url)
                    for link in links:
                        if link not in self.visited_urls and link not in self.queue:
                            self.queue.append(link)

                    page_count += 1

                    # Print status every 10 pages
                    if page_count % 10 == 0:
                        print(
                            f"Processed {page_count} pages. Queue size: {len(self.queue)}. Visited: {len(self.visited_urls)}")

                else:
                    print(f"Failed to fetch {current_url}, status code: {response.status_code}")

            except Exception as e:
                print(f"Error processing {current_url}: {e}")

        print(f"Crawling complete. Processed {page_count} pages.")
        return page_count

    def prepare_for_rag(self, chunk_size=500, chunk_overlap=50):
        """
        Process the scraped data into chunks suitable for RAG
        Returns a list of text chunks with their metadata
        """
        chunks = []
        doc_id = 0

        # Read all scraped data
        with open(self.results_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc_id += 1
                document = json.loads(line)
                url = document["url"]
                content = document["content"]

                # Create a combined text representation
                text_parts = []

                if content["title"]:
                    text_parts.append(f"Title: {content['title']}")

                if content["meta_description"]:
                    text_parts.append(f"Description: {content['meta_description']}")

                for heading in content["headings"]:
                    text_parts.append(f"{heading['level'].upper()}: {heading['text']}")

                for paragraph in content["paragraphs"]:
                    text_parts.append(paragraph)

                for lst in content["lists"]:
                    items_text = ". ".join(lst["items"])
                    text_parts.append(f"List: {items_text}")

                # Add table data
                for i, table in enumerate(content.get("tables", [])):
                    table_text = []
                    table_text.append(f"Table {i+1}:")

                    # Add headers if available
                    if table["headers"]:
                        table_text.append("Headers: " + " | ".join(table["headers"]))

                    # Add rows
                    for row in table["rows"]:
                        table_text.append("Row: " + " | ".join(row))

                    text_parts.append("\n".join(table_text))

                # Combine all text
                full_text = "\n\n".join(text_parts)

                # Skip empty content
                if not full_text.strip():
                    continue

                # Create chunks with overlap
                words = full_text.split()

                for i in range(0, len(words), chunk_size - chunk_overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)

                    chunk = {
                        "doc_id": f"{doc_id}-{i // chunk_size}",
                        "url": url,
                        "text": chunk_text,
                        "source": "walden_university",
                        "url_path": content.get("url_path", "")
                    }

                    chunks.append(chunk)

        # Save chunks to a file
        chunks_file = os.path.join(self.output_folder, "rag_chunks.jsonl")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')

        print(f"Created {len(chunks)} chunks for RAG. Saved to {chunks_file}")
        return chunks

    def save_sitemap(self):
        """Save all discovered URLs to a sitemap file"""
        sitemap_file = os.path.join(self.output_folder, "walden_sitemap.txt")

        with open(sitemap_file, 'w', encoding='utf-8') as f:
            for url in sorted(self.visited_urls):
                f.write(f"{url}\n")

        print(f"Saved {len(self.visited_urls)} URLs to sitemap: {sitemap_file}")


def main():
    # Initialize the scraper
    scraper = WaldenUniversityScraper(base_url="https://www.waldenu.edu/", output_folder="walden_data")

    # Crawl the website (limit to 1000 pages for example)
    scraper.crawl(max_pages=1000)

    # Save all discovered URLs to a sitemap
    scraper.save_sitemap()

    # Prepare data for RAG
    chunks = scraper.prepare_for_rag(chunk_size=500, chunk_overlap=50)

    print(f"Scraping complete. Total pages: {len(scraper.visited_urls)}")
    print(f"Total chunks for RAG: {len(chunks)}")
    print(f"Data saved to: {scraper.output_folder}")


if __name__ == "__main__":
    main()