#!/usr/bin/env python3
"""
Minimal test to check if Wikipedia search returns observations
"""
import requests
from bs4 import BeautifulSoup

def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception as e:
        print(f"[DEBUG] clean_str failed: {e}")
        return p

# Test a simple search
entity = "Irene Jacob"
entity_ = entity.replace(" ", "+")
search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"

print(f"Searching: {search_url}")
response = requests.get(search_url, timeout=10)
print(f"Response status: {response.status_code}")
print(f"Response length: {len(response.text)}")

soup = BeautifulSoup(response.text, features="html.parser")

# Check for search results page
result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
print(f"Search result divs found: {len(result_divs)}")

# Check for page content
paragraphs = soup.find_all("p")
lists = soup.find_all("ul")
print(f"Paragraphs found: {len(paragraphs)}")
print(f"Lists found: {len(lists)}")

# Get text content
page_elements = [p.get_text().strip() for p in paragraphs + lists]
print(f"\nTotal page elements: {len(page_elements)}")

# Filter and show first few
substantial = [p for p in page_elements if len(p.split(" ")) > 2]
print(f"Substantial elements (>2 words): {len(substantial)}")

if substantial:
    print(f"\nFirst substantial element:")
    print(f"{substantial[0][:200]}...")
else:
    print("\nNo substantial elements found!")
    print("\nAll elements:")
    for i, elem in enumerate(page_elements[:5]):
        print(f"{i}: {elem[:100]}")
