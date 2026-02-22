#!/usr/bin/env python3
"""Shorten URLs using TinyURL API (free, no key needed)."""
import sys
import urllib.request
import urllib.parse

def shorten(url: str) -> str:
    api = f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(url, safe='')}"
    req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode().strip()

if __name__ == "__main__":
    for url in sys.argv[1:]:
        print(shorten(url))
