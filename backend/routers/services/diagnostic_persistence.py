"""Fit diagnostics to the known MySQL column capacity without changing API text."""

import re


SHORTENED_SUFFIX = " … [details shortened for storage]"
MYSQL_TEXT_BYTE_LIMITS = {
    "tinytext": 255,
    "text": 65535,
    "mediumtext": 16777215,
    "longtext": 4294967295,
}


def confidence_reason_value(reason, column_type="varchar(255)"):
    """Keep full text when it fits; make any storage shortening explicit.

    VARCHAR/CHAR limits count characters. MySQL TEXT limits count bytes, so a
    multibyte diagnostic must be bounded by its UTF-8 size rather than len().
    An unknown schema uses the previous 255-character capacity conservatively.
    """
    if reason is None:
        return None
    value = str(reason)
    kind = str(column_type or "varchar(255)").strip().lower()
    characters = re.fullmatch(r"(?:var)?char\((\d+)\)", kind)
    byte_limit = MYSQL_TEXT_BYTE_LIMITS.get(kind)
    if byte_limit is not None:
        encoded = value.encode("utf-8")
        if len(encoded) <= byte_limit:
            return value
        suffix = SHORTENED_SUFFIX.encode("utf-8")
        prefix = encoded[:max(0, byte_limit - len(suffix))].decode("utf-8", errors="ignore")
        return prefix + SHORTENED_SUFFIX
    limit = int(characters[1]) if characters else 255
    if len(value) <= limit:
        return value
    if limit < len(SHORTENED_SUFFIX):
        return value[:max(0, limit - 1)] + ("…" if limit else "")
    return value[:limit - len(SHORTENED_SUFFIX)] + SHORTENED_SUFFIX
