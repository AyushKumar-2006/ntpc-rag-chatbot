
import asyncio
import json
import os
from playwright.async_api import async_playwright

PAGES_TO_SCRAPE = [
    "https://www.ntpc.co.in/en",
    "https://www.ntpc.co.in/en/about-us",
    "https://www.ntpc.co.in/en/sustainability",
    "https://www.ntpc.co.in/en/sustainability/approach",
    "https://www.ntpc.co.in/en/sustainability/strategies",
    "https://www.ntpc.co.in/en/sustainability/policies",
    "https://www.ntpc.co.in/en/sustainability/governance",
    "https://www.ntpc.co.in/en/sustainability/reports-publications",
    "https://www.ntpc.co.in/en/sustainability/esg-disclosure",
    "https://www.ntpc.co.in/en/investors",
    "https://www.ntpc.co.in/en/join-us",
    "https://www.ntpc.co.in/en/media",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def clean(text):
    return " ".join(text.split()).strip()

async def scrape_page(page, url):
    print(f"Scraping: {url}")
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)

        title = await page.title()

        paragraphs = []
        for el in await page.query_selector_all("p"):
            t = clean(await el.inner_text())
            if len(t) > 50:
                paragraphs.append(t)

        headings = []
        for el in await page.query_selector_all("h1, h2, h3"):
            t = clean(await el.inner_text())
            if t:
                headings.append(t)

        list_items = []
        for el in await page.query_selector_all("li"):
            t = clean(await el.inner_text())
            if len(t) > 20:
                list_items.append(t)

        from urllib.parse import urlparse
        path_parts = urlparse(url).path.strip("/").split("/")
        section = " - ".join(p.replace("-", " ").title() for p in path_parts if p and p != "en")

        print(f"  Got {len(paragraphs)} paragraphs, {len(headings)} headings")
        return {
            "url": url,
            "title": title,
            "section": section or "General",
            "headings": headings,
            "paragraphs": paragraphs,
            "list_items": list_items,
            "pdf_links": []
        }
    except Exception as e:
        print(f"  Failed: {e}")
        return None

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_pages = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for url in PAGES_TO_SCRAPE:
            data = await scrape_page(page, url)
            if data:
                all_pages.append(data)
            await asyncio.sleep(2)

        await browser.close()

    with open(os.path.join(OUTPUT_DIR, "pages.json"), "w") as f:
        json.dump(all_pages, f, indent=2)

    print(f"\nDone! {len(all_pages)} pages scraped!")
    print("Next: run indexer.py")

if __name__ == "__main__":
    asyncio.run(main())