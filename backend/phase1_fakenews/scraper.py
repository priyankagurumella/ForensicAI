import requests
from bs4 import BeautifulSoup
import re

def scrape_article(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = ''
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)

        # Extract paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Clean
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 100:
            return {"error": "Could not extract enough text from URL"}

        return {
            "title": title,
            "text": text[:5000],  # limit to 5000 chars
            "url": url,
            "char_count": len(text)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test with a real news URL
    url = "https://www.bbc.com/news/world"
    result = scrape_article(url)
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Text preview: {result.get('text', '')[:200]}")