"""
api/routers/preview.py
───────────────────────
POST /preview   — Secure Website Preview using Playwright
"""

import time
import base64
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from playwright.sync_api import sync_playwright

from api.schemas import PreviewRequest, PreviewResponse
from api.security import require_auth
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/preview", tags=["Preview"])

def _generate_preview_sync(url: str) -> bytes:
    with sync_playwright() as p:
        # We use headless=True to hide the browser UI.
        browser = p.chromium.launch(headless=True)
        
        # Create a context with JS enabled so modern SPAs render correctly,
        # but sandboxed by Playwright.
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        page = context.new_page()
        
        # Try to navigate, give it max 10 seconds.
        try:
            # Wait until network is idle or 10s passed.
            page.goto(url, timeout=10000, wait_until="load")
        except Exception as e:
            logger.warning(f"Timeout or error navigating to {url}: {e}")
            # We catch this so we can still screenshot whatever partially loaded.
        
        # Evaluate script to highlight suspicious elements
        highlight_script = """
        () => {
            try {
                // Highlight password fields
                const passwordInputs = document.querySelectorAll('input[type="password"]');
                passwordInputs.forEach(el => {
                    el.style.border = '4px solid red';
                    el.style.boxShadow = '0 0 15px red';
                    el.style.backgroundColor = '#ffcccc';
                });

                // Highlight suspicious iframes
                const iframes = document.querySelectorAll('iframe');
                iframes.forEach(el => {
                    el.style.border = '4px solid orange';
                });

                // Highlight external scripts
                const scripts = document.querySelectorAll('script[src]');
                const currentHost = window.location.hostname;
                let externalCount = 0;
                scripts.forEach(el => {
                    try {
                        const srcUrl = new URL(el.src);
                        if (srcUrl.hostname !== currentHost && srcUrl.hostname !== "") {
                            externalCount++;
                        }
                    } catch(e) {}
                });

                // Add an overlay if suspicious elements are found
                if (passwordInputs.length > 0 || externalCount > 0) {
                    const overlay = document.createElement('div');
                    overlay.style.position = 'fixed';
                    overlay.style.top = '10px';
                    overlay.style.right = '10px';
                    overlay.style.backgroundColor = 'rgba(255, 0, 0, 0.9)';
                    overlay.style.color = 'white';
                    overlay.style.padding = '15px 20px';
                    overlay.style.fontFamily = 'monospace';
                    overlay.style.fontWeight = 'bold';
                    overlay.style.fontSize = '16px';
                    overlay.style.zIndex = '2147483647'; // Max z-index
                    overlay.style.borderRadius = '5px';
                    overlay.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
                    
                    let html = '🚨 SECURITY WARNING 🚨<br>';
                    if (passwordInputs.length > 0) html += `- Found ${passwordInputs.length} Password Field(s)<br>`;
                    if (externalCount > 0) html += `- Found ${externalCount} External Script(s)<br>`;
                    
                    overlay.innerHTML = html;
                    document.body.appendChild(overlay);
                }
            } catch(e) {}
        }
        """
        try:
            page.evaluate(highlight_script)
            # Give a small delay for CSS to apply and animations to settle
            page.wait_for_timeout(500)
        except Exception as e:
            logger.warning(f"Error evaluating highlight script: {e}")
            
        # Take screenshot.
        screenshot_bytes = page.screenshot(full_page=False)
        browser.close()
        return screenshot_bytes

@router.post(
    "",
    response_model=PreviewResponse,
    summary="Secure Website Preview",
    description="Fetches a URL and returns a sandboxed, highlighted screenshot"
)
async def get_preview(
    req: PreviewRequest,
    user: str = Depends(require_auth),
) -> PreviewResponse:
    t0 = time.perf_counter()
    url = req.url
    
    try:
        screenshot_bytes = await run_in_threadpool(_generate_preview_sync, url)
        
        image_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        processing_time = (time.perf_counter() - t0) * 1000
        
        return PreviewResponse(
            image_b64=image_b64,
            processing_time_ms=processing_time
        )
            
    except Exception as e:
        logger.error(f"Error generating preview for {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate preview: {str(e)}"
        )
