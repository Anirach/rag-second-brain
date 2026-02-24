#!/usr/bin/env python3
"""Generate images using Nano Banana Pro via Google AI Studio"""
import requests
import base64
import sys
import json

GOOGLE_API_KEY = "AIzaSyCNTELmQROMXC67W115YevDvKZmx0t-NpM"

def generate_image(prompt: str, output_path: str) -> bool:
    """Generate an image using Gemini 3 Pro Image (Nano Banana Pro)"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={GOOGLE_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Generate an image: {prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        # Extract image from response
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "inlineData" in part:
                    image_data = part["inlineData"]["data"]
                    mime_type = part["inlineData"].get("mimeType", "image/png")
                    image_bytes = base64.b64decode(image_data)
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    print(f"✅ Saved: {output_path} ({mime_type})")
                    return True
        
        print(f"❌ No image in response")
        print(f"Response: {json.dumps(data, indent=2)[:1000]}")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text[:500]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 nano_banana_google.py '<prompt>' <output_path>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    success = generate_image(prompt, output_path)
    sys.exit(0 if success else 1)
