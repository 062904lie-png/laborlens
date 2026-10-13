"""Conservative extraction review signals; never substitutes letters for numbers."""
import re


def extraction_issues(text):
    text = str(text or '')
    issues = []
    if '\ufffd' in text or re.search(r'Ã[\x80-\xff]|â€[\x80-\xff]|ï¿½', text):
        issues.append('encoding_damage')
    # A currency prefix followed by a letter-containing number is suspect.
    # Ordinary prose such as "PHP is ..." and occupational codes are not amounts.
    currency_tokens = re.findall(r'(?:₱\s*|\bPHP\s*)([0-9GOSIl|,]+(?:\.\d{2})?)(?!\w)', text, re.I)
    if any(re.search(r'[GOSIl|]', t, re.I) and (re.search(r'\d', t) or re.fullmatch(r'[SG][Oo]{2,}', t, re.I))
           for t in currency_tokens):
        issues.append('suspicious_currency_ocr')
    tokens = re.findall(r'\S+', text)
    damaged = sum(bool(re.search(r'[a-z][0-9]|[0-9][a-z]', t, re.I)) for t in tokens)
    # Short alphanumeric table IDs alone do not signal damage. Rotated OCR
    # often mixes letters and digits inside long malformed words as well.
    long_damage = sum(len(t) >= 7 and bool(re.search(r'[a-z][0-9]|[0-9][a-z]', t, re.I))
                      and not re.fullmatch(r'[A-Z0-9-]+', t) for t in tokens)
    if len(tokens) > 40 and damaged / len(tokens) > .20 and long_damage >= 3:
        issues.append('heavy_mixed_character_ocr')
    if len(tokens) > 40 and sum(bool(re.search(r'[a-z][^\w\s\-’\',.:;/()][a-z]', t, re.I))
                                for t in tokens) / len(tokens) > .08:
        issues.append('garbled_word_shapes')
    # A Windows source path printed in a document footer is not broken OCR.
    elif len(text) > 250 and len(re.findall(r'[a-z][\^\\][a-z]',
             re.sub(r'[A-Za-z]:\\[^\r\n]+', '', text), re.I)) >= 3:
        issues.append('garbled_word_shapes')
    letters = re.findall(r'\b[a-zA-Z]+\b', text)
    common = sum(w.casefold() in {'the', 'and', 'of', 'to', 'in', 'for', 'shall', 'is', 'workers'}
                 for w in letters)
    if (len(letters) > 80 and sum(len(w) == 1 for w in letters) / len(letters) > .25
            and common / len(letters) < .01):
        issues.append('broken_token_layout')
    return issues
