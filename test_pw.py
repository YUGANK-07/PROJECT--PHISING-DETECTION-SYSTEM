import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        await browser.close()
        print('Playwright works!')

asyncio.run(run())
