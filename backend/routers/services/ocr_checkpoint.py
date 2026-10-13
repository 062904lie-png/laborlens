"""Reusable OCR page results, invalidated whenever the source or OCR profile changes."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


class OCRCheckpoint:
    def __init__(self, filepath, profile):
        source = Path(filepath)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(profile.encode("utf-8"))
        self.directory = source.parent / ".ocr-cache" / digest.hexdigest()

    def read(self, page):
        try:
            value = json.loads((self.directory / f"{page}.json").read_text(encoding="utf-8"))
            return value["text"] if isinstance(value.get("text"), str) else None
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def write(self, page, text):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as stream:
                temporary = stream.name
                json.dump({"text": text}, stream, ensure_ascii=False)
            os.replace(temporary, self.directory / f"{page}.json")
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
