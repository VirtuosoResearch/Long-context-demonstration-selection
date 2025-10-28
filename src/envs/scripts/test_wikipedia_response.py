#!/usr/bin/env python3
"""
Debug script to see what Wikipedia actually returns
"""
import requests

entity = "Albert Einstein"
entity_ = entity.replace(" ", "+")
search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"

# Add User-Agent header
headers = {
    'User-Agent': 'WikiEnvBot/1.0 (Educational Research Project; Python/requests)'
}

print(f"URL: {search_url}")
print("="*60)

response = requests.get(search_url, headers=headers, timeout=10)
print(f"Status Code: {response.status_code}")
print(f"Response Length: {len(response.text)}")
print(f"Final URL (after redirects): {response.url}")
print(f"Number of redirects: {len(response.history)}")

print("\n" + "="*60)
print("RESPONSE CONTENT:")
print("="*60)
print(response.text)
print("\n" + "="*60)
print("RESPONSE HEADERS:")
print("="*60)
for key, value in response.headers.items():
    print(f"{key}: {value}")
