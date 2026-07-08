import asyncio
from features.visual_similarity import detect_visual_similarity_sync
import sys
import logging

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_visual.py <input_url> <reference_url>")
        print("Running default test...")
        input_url = "https://example.com"
        reference_url = "https://example.org"
    else:
        input_url = sys.argv[1]
        reference_url = sys.argv[2]
        
    print(f"Testing visual similarity between {input_url} and {reference_url}")
    
    score, flag, conf = detect_visual_similarity_sync(input_url, reference_url)
    
    print(f"Similarity Score: {score:.4f}")
    print(f"Phishing Flag: {flag}")
    print(f"Confidence: {conf:.4f}")

if __name__ == "__main__":
    main()
