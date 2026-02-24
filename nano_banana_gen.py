#!/usr/bin/env python3
"""Generate images using Nano Banana Pro (Gemini 3 Pro Image) via OpenRouter"""
import requests
import base64
import os
import sys
import json

OPENROUTER_API_KEY = "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"
MODEL = "google/gemini-3-pro-image-preview"

def generate_image(prompt: str, output_path: str) -> bool:
    """Generate an image using Nano Banana Pro"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://clawdbot.com",
        "X-Title": "ClawdBot Image Gen"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Generate an image: {prompt}"
            }
        ],
        "modalities": ["image", "text"],
        "max_tokens": 4096
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract image from response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", [])
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        # Extract base64 data
                        base64_data = image_url.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        with open(output_path, "wb") as f:
                            f.write(image_bytes)
                        print(f"✅ Saved: {output_path}")
                        return True
        
        print(f"❌ No image in response for: {prompt[:50]}...")
        print(f"Response: {json.dumps(data, indent=2)[:500]}")
        return False
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 nano_banana_gen.py <prompt> <output_path>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    success = generate_image(prompt, output_path)
    sys.exit(0 if success else 1)
