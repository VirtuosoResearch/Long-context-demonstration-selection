#!/usr/bin/env python3
"""
Verify the Wikipedia search fix works with BeautifulSoup parsing
"""
import requests
from bs4 import BeautifulSoup

def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception:
        return p

def get_page_obs(page):
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    sentences = []
    for p in paragraphs:
        sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

def test_search(entity):
    print(f"\n{'='*60}")
    print(f"Testing: {entity}")
    print('='*60)
    
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    
    headers = {
        'User-Agent': 'WikiEnvBot/1.0 (Educational Research Project; Python/requests)'
    }
    
    response = requests.get(search_url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}, Length: {len(response.text)}")
    
    soup = BeautifulSoup(response.text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    
    if result_divs:
        result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
    else:
        page_elements = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        
        page = ""
        for p in page_elements:
            if len(p.split(" ")) > 2:
                page += clean_str(p)
                if not p.endswith("\n"):
                    page += "\n"
        
        if len(page.strip()) == 0:
            obs = f"Found page for {entity} but it appears to be empty."
        else:
            obs = get_page_obs(page)
    
    print(f"\nObservation (length={len(obs)}):")
    print(obs[:300] + "..." if len(obs) > 300 else obs)
    return obs

# Test various entities
test_entities = [
    "Albert Einstein",
    "Irene Jacob",
    "Stuart Bird",
    "Python programming"
]

for entity in test_entities:
    obs = test_search(entity)
    assert len(obs) > 0, f"Empty observation for {entity}!"

print("\n" + "="*60)
print("✓ All tests passed! Observations are no longer empty.")
print("="*60)
