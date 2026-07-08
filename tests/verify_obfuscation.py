"""tests/verify_obfuscation.py — verify all new obfuscation features fire correctly."""
import sys; sys.path.insert(0, '.')
from features.obfuscation_features import extract_obfuscation_features as obf

TESTS = [
    # (category, expected_to_fire, url)
    ("LEGIT",      False, "https://microsoft.com"),
    ("LEGIT",      False, "https://paypal.com/signin"),
    ("LEGIT",      False, "https://google.com/search?q=hello"),
    ("UNICODE",    True,  "https://m\u0456crosoft.com"),          # Cyrillic і
    ("UNICODE",    True,  "https://payp\u0430l.com"),             # Cyrillic а
    ("UNICODE",    True,  "https://g\u043e\u043egle.com"),        # Cyrillic о
    ("DIGIT-SUB",  True,  "https://g00gle-login-secure.com"),
    ("DIGIT-SUB",  True,  "https://paypa1.com"),
    ("DIGIT-SUB",  True,  "https://micros0ft.com"),
    ("HEX-IP",     True,  "http://0xC0A80001/billing.php"),
    ("DEC-IP",     True,  "http://3232235521/billing.php"),
    ("DBL-ENC",    True,  "http://paypal%2Ecom%2Flogin"),
    ("ENCODED",    True,  "http://secure.xyz/%2Flogin%2F"),
]

print()
print("  Category    Fire?  obf_score  unicode  homoglyph  hex_ip  dec_ip  dbl_enc")
print("  " + "-" * 75)
ok = 0
for cat, expect_fire, url in TESTS:
    f   = obf(url)
    # "fires" if obfuscation_score > 0.2 OR any individual flag is set
    fired = (
        f["obfuscation_score"] > 0.2
        or f["has_unicode_attack"]
        or f["homoglyph_score"] > 0.5
        or f["has_hex_ip"]
        or f["has_decimal_ip"]
        or f["has_double_encoding"]
        or f["encoded_dot_in_host"]
    )
    result = "OK   " if fired == expect_fire else "WRONG"
    if fired == expect_fire:
        ok += 1
    print(
        f"  [{result}] {cat:<10} {'YES' if fired else 'NO '}"
        f"  {f['obfuscation_score']:.3f}     "
        f" {int(f['has_unicode_attack'])}        "
        f" {f['homoglyph_score']:.3f}     "
        f" {int(f['has_hex_ip'])}       "
        f" {int(f['has_decimal_ip'])}       "
        f" {int(f['has_double_encoding'])}"
    )

print()
print(f"  Result: {ok}/{len(TESTS)} ({ok/len(TESTS)*100:.0f}%)")
