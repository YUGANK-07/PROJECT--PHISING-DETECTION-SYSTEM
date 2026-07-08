"""
data/scripts/fetch_seed_dataset.py  (v2 — realistic URL generation)
────────────────────────────────────────────────────────────────────
Generates a balanced seed dataset of phishing and LEGITIMATE URLs.

Key fix: Legitimate URLs now include realistic paths/queries matching
real-world browsing patterns, so the model can't trivially distinguish
legit from phishing via path presence alone.
"""

from __future__ import annotations

import random
import re
import string
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

SEED = 42
random.seed(SEED)

# ── 500 real well-known legitimate domains ────────────────────────────────────
LEGIT_DOMAINS = [
    # Big tech
    "google.com","youtube.com","facebook.com","twitter.com","instagram.com",
    "linkedin.com","microsoft.com","apple.com","amazon.com","netflix.com",
    "github.com","stackoverflow.com","reddit.com","wikipedia.org","yahoo.com",
    "bing.com","office.com","live.com","outlook.com","hotmail.com",
    "paypal.com","ebay.com","walmart.com","target.com","bestbuy.com",
    "adobe.com","dropbox.com","slack.com","zoom.us","notion.so",
    "spotify.com","twitch.tv","discord.com","telegram.org","whatsapp.com",
    "tiktok.com","pinterest.com","tumblr.com","quora.com","medium.com",
    "wordpress.com","blogger.com","wix.com","squarespace.com","weebly.com",
    "shopify.com","etsy.com","alibaba.com","aliexpress.com","taobao.com",
    "jd.com","baidu.com","qq.com","weibo.com","sina.com.cn",
    # Finance & Banking
    "chase.com","wellsfargo.com","bankofamerica.com","citibank.com","capitalone.com",
    "americanexpress.com","discover.com","fidelity.com","schwab.com","vanguard.com",
    "robinhood.com","coinbase.com","stripe.com","square.com","intuit.com",
    "turbotax.com","quickbooks.com","mint.com","creditkarma.com","experian.com",
    # Gov & Education
    "irs.gov","ssa.gov","usa.gov","whitehouse.gov","congress.gov",
    "cdc.gov","fda.gov","fbi.gov","nasa.gov","nih.gov",
    "harvard.edu","mit.edu","stanford.edu","berkeley.edu","yale.edu",
    "oxford.ac.uk","cambridge.org","coursera.org","edx.org","udemy.com",
    "khanacademy.org","duolingo.com","chegg.com","academia.edu","jstor.org",
    # News & Media
    "cnn.com","bbc.com","nytimes.com","theguardian.com","washingtonpost.com",
    "reuters.com","apnews.com","bloomberg.com","forbes.com","wsj.com",
    "techcrunch.com","theverge.com","wired.com","arstechnica.com","engadget.com",
    "pcmag.com","cnet.com","zdnet.com","venturebeat.com","mashable.com",
    # Cloud & Dev
    "aws.amazon.com","cloud.google.com","azure.microsoft.com","digitalocean.com","heroku.com",
    "vercel.com","netlify.com","cloudflare.com","fastly.com","akamai.com",
    "gitlab.com","bitbucket.org","npmjs.com","pypi.org","hub.docker.com",
    "kubernetes.io","terraform.io","ansible.com","jenkins.io","circleci.com",
    # E-commerce & Retail
    "wayfair.com","overstock.com","homedepot.com","lowes.com","ikea.com",
    "zappos.com","nordstrom.com","macys.com","gap.com","zara.com",
    "hm.com","uniqlo.com","forever21.com","shein.com","asos.com",
    # Health
    "webmd.com","mayoclinic.org","healthline.com","medscape.com","drugs.com",
    "cvs.com","walgreens.com","riteaid.com","goodrx.com","zocdoc.com",
    # Travel
    "booking.com","airbnb.com","expedia.com","hotels.com","tripadvisor.com",
    "kayak.com","priceline.com","united.com","delta.com","southwest.com",
    "aa.com","british-airways.com","emirates.com","lufthansa.com","airfrance.com",
    # Misc popular
    "yelp.com","doordash.com","ubereats.com","grubhub.com","instacart.com",
    "uber.com","lyft.com","airbnb.com","vrbo.com","homeaway.com",
    "imdb.com","rotten tomatoes.com","fandango.com","hulu.com","disneyplus.com",
    "hbomax.com","peacocktv.com","paramountplus.com","crunchyroll.com","funimation.com",
    "chess.com","lichess.org","steampowered.com","epicgames.com","origin.com",
    "ea.com","ubisoft.com","blizzard.com","riotgames.com","minecraft.net",
]

# Realistic paths for legitimate sites
LEGIT_PATHS = [
    "/", "/about", "/contact", "/login", "/signin", "/signup", "/register",
    "/account", "/dashboard", "/settings", "/profile", "/home",
    "/products", "/services", "/pricing", "/plans", "/features",
    "/help", "/support", "/faq", "/docs", "/documentation", "/api",
    "/blog", "/news", "/press", "/careers", "/jobs",
    "/terms", "/privacy", "/security", "/legal",
    "/search?q=laptop", "/search?q=shoes", "/search?q=python+tutorial",
    "/en-us/account", "/us/signin", "/us/sign-in",
    "/auth/login", "/auth/signin", "/user/login",
    "/shop/cart", "/checkout", "/order/history",
    "/download", "/get-started", "/free-trial",
    "/s?k=headphones", "/dp/B08N5WRWNW",
    "/watch?v=dQw4w9WgXcQ", "/feed", "/timeline",
    "/issues", "/pull-requests", "/repository",
    "/article/tech-news-today", "/story/latest",
]

LEGIT_QUERY_PARAMS = [
    "", "", "",   # most URLs have no query
    "?ref=homepage", "?utm_source=google", "?lang=en",
    "?page=1", "?tab=overview", "?section=security",
]

# ── Known phishing patterns ────────────────────────────────────────────────────
PHISHING_DOMAINS_TEMPLATES = [
    # Brand + keyword abuse
    "{brand}-secure-{kw}.{tld}", "{brand}-{kw}-verify.{tld}",
    "{kw}-{brand}-login.{tld}", "secure-{brand}-{kw}.{tld}",
    "{brand}-account-{kw}.{tld}", "{brand}.{kw}-{noise}.{tld}",
    "www.{brand}-{kw}.{tld}", "{kw}.{brand}-verify.{tld}",
    # Lookalike
    "{brand}support.{tld}", "{brand}help.{tld}", "{brand}online.{tld}",
    "my{brand}.{tld}", "get{brand}.{tld}", "{brand}secure.{tld}",
    # Subdomain abuse
    "{brand}.{noise}.{tld}", "login.{brand}-{noise}.{tld}",
    "secure.{noise}-{brand}.{tld}", "account.{noise}.{tld}",
    # Random / gibberish
    "{noise}-{noise2}.{tld}", "{kw}{noise}.{tld}",
    "{noise}.{free_tld}", "{noise}-{kw}.{free_tld}",
]

PHISHING_PATHS = [
    "/login.php", "/signin.php", "/verify.php", "/update.php",
    "/secure/login", "/account/verify", "/billing/update",
    "/confirm-identity", "/suspended-account",
    "/wp-content/login.php", "/wp-login.php",
    "/login?redirect=account&token={token}",
    "/verify?email=user@gmail.com&code={token}",
    "/auth/callback?state={token}&session={token2}",
    "/update-billing.php?id={token}",
    "/confirm/{token}/step2",
    "/{token}/{token2}/form.html",
    "/phishing/{noise}/login",
    "/free-prize/claim-now.php",
]

BRANDS = [
    "paypal","amazon","apple","microsoft","google","netflix","facebook",
    "instagram","twitter","wellsfargo","chase","bankofamerica","ebay",
    "dropbox","linkedin","outlook","office365","icloud","samsung","steam",
]

KEYWORDS = [
    "secure","login","verify","confirm","update","account","billing",
    "alert","suspended","access","support","auth","signin","checkout",
    "reward","bonus","payment","invoice","urgent","claim",
]

FREE_TLDS = [".xyz",".tk",".ml",".ga",".cf",".top",".click",".online",
             ".site",".info",".biz",".icu",".live",".shop",".vip"]
RISKY_TLDS = [".xyz",".top",".online",".site",".tk",".ml",".click"]

def _rand_token(n=12):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _rand_noise():
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))

# Typosquat substitution tables
_TYPO_SUBS = [
    # char-substitution
    lambda s: s.replace('o', '0', 1),
    lambda s: s.replace('l', '1', 1),
    lambda s: s.replace('i', '1', 1),
    lambda s: s.replace('a', '4', 1),
    lambda s: s.replace('e', '3', 1),
    lambda s: s.replace('s', '5', 1),
    lambda s: s.replace('g', '9', 1),
    # multi-char confusion
    lambda s: s.replace('m', 'rn', 1),
    lambda s: s.replace('m', 'nn', 1),
    lambda s: s.replace('w', 'vv', 1),
    lambda s: s.replace('d', 'cl', 1),
    # character insertion / deletion
    lambda s: s + s[-1],          # double last char (amazoon)
    lambda s: s[:-1],             # drop last char (amazo)
    lambda s: s[0] + s,           # double first char (aamazon)
    lambda s: s[:3] + s[2:],      # duplicate mid char
]

def _gen_typosquat_url() -> str:
    """Generate a bare typosquat domain (no suspicious path needed)."""
    brand   = random.choice(BRANDS)
    mut     = random.choice(_TYPO_SUBS)
    typo    = mut(brand)
    if typo == brand:          # mutation had no effect, add digit
        typo = brand[:-1] + random.choice('0123456789')
    tld = random.choice([".com", ".net", ".org", ".co", ".io"])
    # Sometimes add a minimal path to look more convincing
    path = random.choice(["", "/", "/login", "/signin", "/account", "/verify"])
    scheme = random.choice(["http", "https"])
    return f"{scheme}://{typo}{tld}{path}"

# ── Unicode confusable substitutions ─────────────────────────────────────────
# Cyrillic / Greek chars that look identical to Latin
_CONFUSABLE_MAP = {
    "a": "\u0430",   # Cyrillic а
    "e": "\u0435",   # Cyrillic е
    "o": "\u043e",   # Cyrillic о
    "p": "\u0440",   # Cyrillic р
    "c": "\u0441",   # Cyrillic с
    "x": "\u0445",   # Cyrillic х
    "i": "\u0456",   # Ukrainian і
    "O": "\u039f",   # Greek Ο
    "A": "\u0391",   # Greek Α
}

def _gen_unicode_homograph_url() -> str:
    """Replace one Latin char in a brand name with a confusable Unicode char."""
    brand = random.choice(BRANDS)
    # Find substitutable chars
    candidates = [(i, c) for i, c in enumerate(brand) if c in _CONFUSABLE_MAP]
    if not candidates:
        # fallback: just use a digit sub
        return _gen_typosquat_url()
    idx, ch = random.choice(candidates)
    spoofed  = brand[:idx] + _CONFUSABLE_MAP[ch] + brand[idx+1:]
    tld  = random.choice([".com", ".net", ".org"])
    path = random.choice(["", "/login", "/signin", "/account", "/verify"])
    return f"http://{spoofed}{tld}{path}"


def _gen_encoded_url() -> str:
    """Generate URL with URL-encoding obfuscation tricks."""
    brand = random.choice(BRANDS)
    path  = random.choice(PHISHING_PATHS).format(
        token=_rand_token(16), token2=_rand_token(8), noise=_rand_noise()
    )
    tricks = [
        # Double-encoded path
        lambda: f"http://{brand}-secure.com{path.replace('/', '%2F')}",
        # Encoded dot in hostname
        lambda: f"http://{brand}%2Elogin%2Ecom{path}",
        # Hex IP
        lambda: f"http://0x{random.randint(0xC0000001,0xDFFFFFFF):08x}{path}",
        # Decimal IP
        lambda: f"http://{random.randint(3000000000,3500000000)}{path}",
        # @-redirect trick
        lambda: f"http://{brand}.com@{_rand_noise()}.{random.choice(['xyz','top','click'])}{path}",
        # Double slash redirect
        lambda: f"http://{brand}-{random.choice(KEYWORDS)}.com//{path}",
        # Percent-encoded brand in path
        lambda: f"http://{_rand_noise()}.com/%7B{brand}%7D/login",
    ]
    return random.choice(tricks)()


def _gen_alt_ip_url() -> str:
    """Generate URLs using alternative IP representations."""
    path = random.choice(PHISHING_PATHS).format(
        token=_rand_token(12), token2=_rand_token(8), noise=_rand_noise()
    )
    a, b, c, d = (random.randint(1,254) for _ in range(4))
    tricks = [
        f"http://0x{a:02x}{b:02x}{c:02x}{d:02x}{path}",                       # hex IP
        f"http://{a*16777216 + b*65536 + c*256 + d}{path}",                    # decimal IP
        f"http://0{a:03o}.0{b:03o}.0{c:03o}.0{d:03o}{path}",                  # octal IP
        f"http://{a}.{b}.{c}.{d}{path}",                                        # plain IP
        f"http://[::ffff:{a}.{b}.{c}.{d}]{path}",                              # IPv6 mapped
    ]
    return random.choice(tricks)


def _gen_legit_url() -> str:
    domain = random.choice(LEGIT_DOMAINS)
    scheme = "https" if random.random() > 0.05 else "http"
    path   = random.choice(LEGIT_PATHS)
    qp     = random.choice(LEGIT_QUERY_PARAMS)
    return f"{scheme}://{domain}{path}{qp}"

def _gen_phishing_url() -> str:
    template = random.choice(PHISHING_DOMAINS_TEMPLATES)
    brand    = random.choice(BRANDS)
    kw       = random.choice(KEYWORDS)
    noise    = _rand_noise()
    noise2   = _rand_noise()
    free_tld = random.choice(FREE_TLDS)
    risky_tld = random.choice(RISKY_TLDS)

    domain = template.format(
        brand=brand, kw=kw, noise=noise, noise2=noise2,
        tld=risky_tld.lstrip("."), free_tld=free_tld.lstrip(".")
    )

    path = random.choice(PHISHING_PATHS).format(
        token=_rand_token(16), token2=_rand_token(8), noise=noise
    )

    # Sometimes use IP as host
    if random.random() < 0.08:
        ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
        return f"http://{ip}{path}"

    # Sometimes add @ redirect trick
    if random.random() < 0.05:
        return f"http://{domain}@{_rand_token(8)}.{random.choice(['com','net','org'])}{path}"

    scheme = "http" if random.random() > 0.35 else "https"
    return f"{scheme}://{domain}{path}"


def generate_seed_dataset(
    n_legit: int = 75_000,
    n_phish: int = 75_000,
    output_dir: Path = Path("data/raw"),
) -> Path:
    """Generate a balanced seed dataset covering all attack categories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "seed_dataset.csv"

    # Phishing budget split across 5 attack categories
    n_complex   = int(n_phish * 0.50)   # 50% — keyword/brand domain combos
    n_typo      = int(n_phish * 0.15)   # 15% — char substitution typosquats
    n_unicode   = int(n_phish * 0.15)   # 15% — Unicode confusable / homograph
    n_encoded   = int(n_phish * 0.12)   # 12% — URL-encoding obfuscation tricks
    n_altip     = n_phish - n_complex - n_typo - n_unicode - n_encoded  # 8% alt-IP

    print(
        f"Generating {n_legit:,} legit + "
        f"{n_complex:,} complex + {n_typo:,} typosquats + "
        f"{n_unicode:,} Unicode + {n_encoded:,} encoded + {n_altip:,} alt-IP …"
    )

    legit_urls   = [_gen_legit_url()             for _ in range(n_legit)]
    complex_urls = [_gen_phishing_url()          for _ in range(n_complex)]
    typo_urls    = [_gen_typosquat_url()         for _ in range(n_typo)]
    unicode_urls = [_gen_unicode_homograph_url() for _ in range(n_unicode)]
    encoded_urls = [_gen_encoded_url()           for _ in range(n_encoded)]
    altip_urls   = [_gen_alt_ip_url()            for _ in range(n_altip)]

    phish_urls   = complex_urls + typo_urls + unicode_urls + encoded_urls + altip_urls
    sources      = (
        ["seed_legit"]   * n_legit
        + ["seed_phish"]   * n_complex
        + ["seed_typo"]    * n_typo
        + ["seed_unicode"] * n_unicode
        + ["seed_encoded"] * n_encoded
        + ["seed_altip"]   * n_altip
    )

    df = pd.DataFrame({
        "url":    legit_urls + phish_urls,
        "label":  [0] * n_legit + [1] * len(phish_urls),
        "source": sources,
    }).sample(frac=1, random_state=SEED).reset_index(drop=True)

    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")
    return out


if __name__ == "__main__":
    generate_seed_dataset()
