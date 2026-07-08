import asyncio
import io
import time
from typing import Tuple

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import imagehash
from playwright.async_api import async_playwright

from utils.logger import get_logger

logger = get_logger(__name__)

# Cache the model to avoid reloading
_resnet_model = None
_resnet_transforms = None

def get_resnet_model():
    global _resnet_model, _resnet_transforms
    if _resnet_model is None:
        logger.info("Loading ResNet50 model for visual similarity...")
        # Load pre-trained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove final fully connected layer to get 2048-dim feature vector
        _resnet_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        _resnet_model.eval()

        _resnet_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _resnet_model, _resnet_transforms

async def capture_screenshot(url: str, timeout: int = 15000) -> Image.Image | None:
    """Captures a screenshot of the given URL using headless Playwright."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout)
            except Exception as e:
                logger.warning(f"Timeout or error navigating to {url}: {e}")
                # Sometimes it loads partially, let's try to capture anyway
            
            # Wait a bit for animations/rendering
            await asyncio.sleep(1.0)
            
            screenshot_bytes = await page.screenshot(full_page=False)
            await browser.close()
            
            img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
            return img
    except Exception as e:
        logger.error(f"Failed to capture screenshot for {url}: {e}")
        return None

def extract_deep_features(image: Image.Image) -> torch.Tensor:
    """Extracts CNN features from a given image using ResNet50."""
    model, transform = get_resnet_model()
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img_tensor)
    return features.flatten()

def compute_phash(image: Image.Image) -> imagehash.ImageHash:
    """Computes the perceptual hash of an image."""
    return imagehash.phash(image)

async def detect_visual_similarity(
    input_url: str, 
    reference_url: str, 
    threshold: float = 0.85
) -> Tuple[float, bool, float]:
    """
    Compares the visual similarity of two URLs.
    
    Returns:
        similarity_score: Float between 0.0 and 1.0
        phishing_flag: Boolean (True if similarity > threshold)
        confidence: Float confidence score of the prediction
    """
    logger.info(f"Comparing visual similarity: {input_url} vs {reference_url}")
    
    # Capture both screenshots concurrently
    input_img_task = asyncio.create_task(capture_screenshot(input_url))
    ref_img_task = asyncio.create_task(capture_screenshot(reference_url))
    
    input_img, ref_img = await asyncio.gather(input_img_task, ref_img_task)
    
    if input_img is None or ref_img is None:
        logger.warning("Could not capture one or both screenshots for visual similarity.")
        return 0.0, False, 0.0
        
    # 1. Deep Feature Similarity (CNN)
    input_features = extract_deep_features(input_img)
    ref_features = extract_deep_features(ref_img)
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(input_features, ref_features, dim=0).item()
    
    # 2. Perceptual Hashing Similarity
    input_hash = compute_phash(input_img)
    ref_hash = compute_phash(ref_img)
    
    # Hash difference is 0 to 64. Convert to similarity 0.0 to 1.0.
    hash_diff = input_hash - ref_hash
    phash_sim = max(0.0, 1.0 - (hash_diff / 64.0))
    
    # Combine scores (70% CNN, 30% pHash)
    similarity_score = (0.7 * cos_sim) + (0.3 * phash_sim)
    
    # Determine flag and confidence
    phishing_flag = bool(similarity_score >= threshold)
    
    # Confidence scales based on how far it is from the threshold
    # Range [0, 1] mapped from distance to threshold
    if phishing_flag:
        confidence = min(1.0, (similarity_score - threshold) / (1.0 - threshold) * 0.5 + 0.5)
    else:
        confidence = min(1.0, (threshold - similarity_score) / threshold * 0.5 + 0.5)
        
    logger.info(f"Visual Similarity Score: {similarity_score:.4f} (CNN: {cos_sim:.4f}, pHash: {phash_sim:.4f})")
    
    return similarity_score, phishing_flag, float(confidence)

def detect_visual_similarity_sync(input_url: str, reference_url: str, threshold: float = 0.85) -> Tuple[float, bool, float]:
    """Synchronous wrapper for detect_visual_similarity."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, detect_visual_similarity(input_url, reference_url, threshold))
                return future.result()
        else:
            return loop.run_until_complete(detect_visual_similarity(input_url, reference_url, threshold))
    except Exception:
        return asyncio.run(detect_visual_similarity(input_url, reference_url, threshold))
