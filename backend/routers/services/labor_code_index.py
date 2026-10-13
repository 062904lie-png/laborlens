"""Structured lookups for the bundled Labor Code document."""

from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from docx import Document as DocxDocument

from routers.services.document_text_cleaner import strip_document_footnotes


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = BACKEND_DIR.parent
LABOR_CODE_FILENAME = "Labor-Code-of-the-Philippines-Renumbered-2022-2.docx"
LABOR_CODE_SOURCE_URL = "https://dole.gov.ph/labor-code-of-the-philippines-2/"
DISPLAY_DASH = "\u2013"

ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def roman_to_int(value: str) -> Optional[int]:
    total = 0
    prev = 0
    for char in reversed(value.upper()):
        current = ROMAN_VALUES.get(char)
        if current is None:
            return None
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total or None


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\u2010-\u2015-]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_article_title(rest: str) -> str:
    title_part = re.split(r"\s+[\u2013\u2014-]\s+", rest, maxsplit=1)[0]
    title_part = re.sub(r"\.\d+\s*$", "", title_part).strip()
    return title_part.rstrip(".").strip()


def split_article_title_body(rest: str) -> Tuple[str, str]:
    parts = re.split(r"\s+[\u2013\u2014-]\s+", rest, maxsplit=1)
    title = clean_article_title(parts[0])
    body = parts[1].strip() if len(parts) > 1 else ""
    return title, body


def clean_chapter_title(title: str) -> str:
    title = title.strip()
    return re.sub(r"\d+\s*$", "", title).strip()


def should_append_article_content(paragraph: str, last_was_blank: bool) -> bool:
    if re.fullmatch(r"\d+", paragraph):
        return False
    if last_was_blank and re.match(r"^\d+\s+", paragraph):
        return False
    return True


def display_source_name(path: Path) -> str:
    name = path.name
    if name.endswith(LABOR_CODE_FILENAME) and name != LABOR_CODE_FILENAME:
        return LABOR_CODE_FILENAME
    return name


def resolve_labor_code_docx() -> Optional[Path]:
    candidates: List[Path] = []
    search_dirs = [
        BACKEND_DIR,
        BACKEND_DIR / "uploads",
        WORKSPACE_DIR,
        WORKSPACE_DIR / "Knowledge Base",
    ]
    patterns = [LABOR_CODE_FILENAME, f"*_{LABOR_CODE_FILENAME}"]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            candidates.extend(path for path in directory.glob(pattern) if path.is_file())
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_article(text: str) -> Optional[Dict[str, Any]]:
    match = re.match(
        r"^ART\.\s*(?P<number>\d+)\.\s*(?:\[[^\]]+\]\s*)?(?P<rest>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    title, body = split_article_title_body(match.group("rest"))
    if not title:
        return None
    return {
        "number": int(match.group("number")),
        "title": title,
        "heading": f"ART. {int(match.group('number'))}. {title}.",
        "content_parts": [body] if body else [],
    }


@lru_cache(maxsize=4)
def _load_labor_code_index_for_path(doc_path: str) -> Dict[str, Any]:
    doc = DocxDocument(doc_path)
    paragraphs = [" ".join(p.text.split()) for p in doc.paragraphs]

    chapters: List[Dict[str, Any]] = []
    current_chapter: Optional[Dict[str, Any]] = None
    current_article: Optional[Dict[str, Any]] = None
    last_was_blank = False
    skipping_footnote = False

    for paragraph in paragraphs:
        if not paragraph:
            last_was_blank = True
            skipping_footnote = False
            continue

        if not strip_document_footnotes(paragraph):
            skipping_footnote = True
            last_was_blank = False
            continue
        if skipping_footnote:
            last_was_blank = False
            continue

        chapter_match = re.match(
            r"^Chapter\s+(?P<roman>[IVXLCDM]+)\s*[\u2013\u2014-]\s*(?P<title>.+)$",
            paragraph,
            flags=re.IGNORECASE,
        )
        if chapter_match:
            roman = chapter_match.group("roman").upper()
            title = clean_chapter_title(chapter_match.group("title"))
            current_chapter = {
                "roman": roman,
                "number": roman_to_int(roman),
                "title": title,
                "heading": f"Chapter {roman} {DISPLAY_DASH} {title}",
                "articles": [],
            }
            chapters.append(current_chapter)
            current_article = None
            last_was_blank = False
            continue

        article = parse_article(paragraph)
        if article and current_chapter:
            article["chapter_heading"] = current_chapter["heading"]
            article["chapter_title"] = current_chapter["title"]
            current_chapter["articles"].append(article)
            current_article = article
            last_was_blank = False
            continue

        if current_article and should_append_article_content(paragraph, last_was_blank):
            current_article["content_parts"].append(paragraph)

        last_was_blank = False

    for chapter in chapters:
        for article in chapter["articles"]:
            article["content"] = strip_document_footnotes(
                "\n\n".join(article.pop("content_parts", []))
            )

    return {
        "source": display_source_name(Path(doc_path)),
        "source_path": doc_path,
        "source_url": LABOR_CODE_SOURCE_URL,
        "chapters": chapters,
    }


def empty_labor_code_index() -> Dict[str, Any]:
    return {
        "source": LABOR_CODE_FILENAME,
        "source_path": "",
        "source_url": LABOR_CODE_SOURCE_URL,
        "chapters": [],
    }


def load_labor_code_index() -> Dict[str, Any]:
    doc_path = resolve_labor_code_docx()
    if not doc_path:
        logger.warning("Labor Code structured lookup file not found. Expected %s", LABOR_CODE_FILENAME)
        return empty_labor_code_index()
    try:
        return _load_labor_code_index_for_path(str(doc_path))
    except Exception:
        logger.exception("Failed to load Labor Code structured lookup file: %s", doc_path)
        return empty_labor_code_index()


def extract_requested_chapter(query: str) -> Optional[Dict[str, Any]]:
    index = load_labor_code_index()
    query_norm = normalize_text(query)

    chapter_match = re.search(
        r"\bchapter\s+(?P<chapter>[ivxlcdm]+|\d+)\b",
        query,
        flags=re.IGNORECASE,
    )
    if chapter_match:
        chapter_value = chapter_match.group("chapter")
        number = int(chapter_value) if chapter_value.isdigit() else roman_to_int(chapter_value)
        candidates = [chapter for chapter in index["chapters"] if chapter["number"] == number]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            query_words = set(query_norm.split())
            scored = [
                (len(query_words & set(normalize_text(chapter["title"]).split())), chapter)
                for chapter in candidates
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > 0:
                return scored[0][1]
            return candidates[0]

    best_chapter = None
    best_score = 0
    query_words = set(query_norm.split())
    for chapter in index["chapters"]:
        title_words = set(normalize_text(chapter["title"]).split())
        score = len(query_words & title_words)
        if score > best_score:
            best_score = score
            best_chapter = chapter
    if best_score >= 3:
        return best_chapter
    return None


def is_chapter_article_list_query(query: str) -> bool:
    query_norm = normalize_text(query)
    if re.search(r"\b(?:article|art)\s*(?:no\s*)?\d+\b", query_norm):
        return False
    if not re.search(r"\barticles?\b", query_norm):
        return False

    query_words = set(query_norm.split())
    list_words = {
        "inside",
        "under",
        "in",
        "included",
        "contains",
        "list",
        "what",
        "which",
        "ano",
        "alin",
        "mga",
        "sa",
        "sulod",
        "sakop",
        "diin",
    }
    has_list_word = bool(query_words & list_words)
    if "chapter" in query_words and has_list_word:
        return True
    return has_list_word and extract_requested_chapter(query) is not None


def extract_requested_article(query: str) -> Optional[Dict[str, Any]]:
    article_match = re.search(
        r"\b(?:art|article)\.?\s*(?:no\.?\s*)?(?P<number>\d+)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not article_match:
        return None

    article_number = int(article_match.group("number"))
    index = load_labor_code_index()
    candidates = [
        article
        for chapter in index["chapters"]
        for article in chapter["articles"]
        if article["number"] == article_number
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    query_words = set(normalize_text(query).split())
    scored = [
        (len(query_words & set(normalize_text(article["title"]).split())), article)
        for article in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def is_article_content_query(query: str) -> bool:
    article = extract_requested_article(query)
    if not article:
        return False
    # A direct Article reference is a valid lookup even when the user writes it
    # as a short follow-up, e.g. "article 91" or "i mean article 91".
    if re.search(r"\b(?:art|article)\.?\s*(?:no\.?\s*)?\d+\b", query, flags=re.IGNORECASE):
        return True

    query_words = set(normalize_text(query).split())
    content_words = {
        "what",
        "ano",
        "content",
        "text",
        "body",
        "meaning",
        "definition",
        "define",
        "explain",
        "say",
        "says",
        "provision",
        "about",
        "of",
        "sa",
    }
    return bool(query_words & content_words) or article["title"].lower() in query.lower()


def build_article_definition(article: Dict[str, Any], language: str) -> str:
    if article["number"] == 36:
        if language == "fil":
            return (
                "Sa madaling sabi, ang **Regulatory Power** sa ART. 36 ay ang kapangyarihan "
                "ng Secretary of Labor na higpitan at i-regulate ang recruitment at "
                "placement activities ng mga agency na sakop ng Title na ito."
            )
        if language == "hil":
            return (
                "Sa simple nga hambal, ang **Regulatory Power** sa ART. 36 amo ang gahum sang "
                "Secretary of Labor nga mag-restrict kag mag-regulate sang recruitment "
                "kag placement activities sang mga agency nga sakop sang sini nga Title."
            )
        return (
            "In simple terms, **Regulatory Power** in ART. 36 is the authority of the "
            "Secretary of Labor to restrict and regulate recruitment and placement "
            "activities of agencies covered by this Title."
        )

    if language == "fil":
        return (
            f"Sa madaling sabi, ang **{article['title']}** ang paksang tinatalakay "
            f"sa **ART. {article['number']}. {article['title']}.** ng Labor Code."
        )
    if language == "hil":
        return (
            f"Sa simple nga hambal, ang **{article['title']}** amo ang topic nga ginatalakay "
            f"sa **ART. {article['number']}. {article['title']}.** sang Labor Code."
        )
    return (
        f"In simple terms, **{article['title']}** is the subject covered by "
        f"**ART. {article['number']}. {article['title']}.** of the Labor Code."
    )


def answer_article_content(query: str, language: str = "en") -> Optional[Dict[str, Any]]:
    if not is_article_content_query(query):
        return None

    index = load_labor_code_index()
    article = extract_requested_article(query)
    if not article:
        return None

    content = strip_document_footnotes(article.get("content", ""))
    if not content:
        return None

    if language == "fil":
        intro = f"Sige. Heto ang nakasaad sa **ART. {article['number']}. {article['title']}.**"
        article_label = "Teksto mula sa document:"
    elif language == "hil":
        intro = f"Sige. Ari ang nasulat sa **ART. {article['number']}. {article['title']}.**"
        article_label = "Teksto halin sa document:"
    else:
        intro = f"Sure. Here is what **ART. {article['number']}. {article['title']}.** says."
        article_label = "Text from the document:"

    answer = "\n\n".join([
        intro,
        build_article_definition(article, language),
        f"{article_label}\n\n{content}",
        (
            "Legal Basis:\n"
            f"- Labor Code of the Philippines, ART. {article['number']}. {article['title']}.\n"
            f"- Source: {index['source']}"
        ),
    ])

    return {
        "answer": answer,
        "sources": [{
            "label": "Source 1",
            "source": index["source"],
            "document_title": index["source"],
            "source_name": "DOLE",
            "source_url": index.get("source_url", LABOR_CODE_SOURCE_URL),
            "law_version": "2022 renumbered",
            "page": "",
            "citation": f"ART. {article['number']}. {article['title']}.",
            "article_number": str(article["number"]),
            "article_title": article["title"],
            "score": 1.0,
        }],
        "intent": "article_content_lookup",
        "kb_found": True,
    }


def first_supported_sentences(text: str, max_sentences: int = 2, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "The knowledge base lists this Article but does not provide additional text."

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text)
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) <= max_chars:
        return summary

    clipped = summary[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "..."


def find_chapter_by_number_and_title(number: Optional[int], title: str = "") -> Optional[Dict[str, Any]]:
    if not number:
        return None

    index = load_labor_code_index()
    candidates = [chapter for chapter in index["chapters"] if chapter["number"] == number]
    if not candidates:
        return None

    title_norm = normalize_text(title)
    if title_norm:
        for chapter in candidates:
            chapter_title_norm = normalize_text(chapter["title"])
            if title_norm == chapter_title_norm or title_norm in chapter_title_norm:
                return chapter
        query_words = set(title_norm.split())
        scored = [
            (len(query_words & set(normalize_text(chapter["title"]).split())), chapter)
            for chapter in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]

    return candidates[0]


def extract_chapter_from_history(history: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, Any]]:
    if not history:
        return None

    chapter_re = re.compile(
        r"Chapter\s+(?P<chapter>[IVXLCDM]+|\d+)\s*(?:[\u2013\u2014-]|\?|â€“|&ndash;)\s*(?P<title>[^\n\r.]+)",
        flags=re.IGNORECASE,
    )

    for message in reversed(history):
        content = message.get("content", "")
        matches = list(chapter_re.finditer(content))
        for match in reversed(matches):
            chapter_value = match.group("chapter")
            number = int(chapter_value) if chapter_value.isdigit() else roman_to_int(chapter_value)
            title = clean_chapter_title(match.group("title"))
            chapter = find_chapter_by_number_and_title(number, title)
            if chapter:
                return chapter
    return None


def is_chapter_explanation_query(query: str) -> bool:
    query_norm = normalize_text(query)
    words = set(query_norm.split())
    explain_words = {
        "explain",
        "explanation",
        "details",
        "detail",
        "meaning",
        "summarize",
        "summary",
        "elaborate",
        "paliwanag",
        "ipaliwanag",
    }
    group_words = {"all", "them", "those", "these", "articles", "article", "each", "lahat", "mga"}
    return bool(words & explain_words) and bool(words & group_words)


def answer_chapter_article_explanations(
    query: str,
    language: str = "en",
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    if not is_chapter_explanation_query(query):
        return None

    index = load_labor_code_index()
    chapter = extract_requested_chapter(query) or extract_chapter_from_history(history)
    if not chapter:
        return None

    explanation_lines = []
    sources = []
    for i, article in enumerate(chapter["articles"], 1):
        explanation = first_supported_sentences(article.get("content", ""))
        explanation_lines.append(
            f"- **ART. {article['number']}. {article['title']}.** {explanation} (Source {i})"
        )
        sources.append({
            "label": f"Source {i}",
            "source": index["source"],
            "document_title": index["source"],
            "source_name": "DOLE",
            "source_url": index.get("source_url", LABOR_CODE_SOURCE_URL),
            "law_version": "2022 renumbered",
            "page": "",
            "citation": f"ART. {article['number']}. {article['title']}.",
            "article_number": str(article["number"]),
            "article_title": article["title"],
            "score": 1.0,
        })

    source_labels = ", ".join(source["label"] for source in sources)
    if language == "fil":
        intro = (
            f"Sige. Narito ang maikling paliwanag ng bawat Article sa "
            f"**{chapter['heading']}**, batay lamang sa Labor Code document:"
        )
        source_text = f"Mga ginamit na source: {source_labels}"
    elif language == "hil":
        intro = (
            f"Sige. Ari ang malip-ot nga pagpaathag sang kada Article sa "
            f"**{chapter['heading']}**, base lamang sa Labor Code document:"
        )
        source_text = f"Gin-gamit nga sources: {source_labels}"
    else:
        intro = (
            f"Sure. Here is a brief explanation of each Article under "
            f"**{chapter['heading']}**, based only on the Labor Code document:"
        )
        source_text = f"Sources used: {source_labels}"

    answer = "\n\n".join([
        intro,
        "\n".join(explanation_lines),
        source_text,
        (
            "Legal Basis:\n"
            f"- Labor Code of the Philippines, {chapter['heading']}\n"
            f"- Source: {index['source']}"
        ),
    ])

    return {
        "answer": answer,
        "sources": sources,
        "intent": "chapter_article_explanation_lookup",
        "kb_found": True,
    }


def answer_labor_code_lookup(
    query: str,
    language: str = "en",
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    from routers.services.named_issuance import requested_issuance
    if requested_issuance(query):
        return None
    try:
        return (
            answer_article_content(query, language)
            or answer_chapter_article_explanations(query, language, history)
            or answer_chapter_article_list(query, language)
            or answer_missing_labor_code_reference(query, language)
        )
    except Exception:
        logger.exception("Structured Labor Code lookup failed for query: %s", query)
        return None


def answer_missing_labor_code_reference(query: str, language: str = "en") -> Optional[Dict[str, Any]]:
    article_match = re.search(
        r"\b(?:art|article)\.?\s*(?:no\.?\s*)?(?P<number>\d+)\b",
        query,
        flags=re.IGNORECASE,
    )
    chapter_match = re.search(
        r"\bchapter\s+(?P<chapter>[ivxlcdm]+|\d+)\b",
        query,
        flags=re.IGNORECASE,
    )

    if article_match and not extract_requested_article(query):
        number = article_match.group("number")
        if language == "fil":
            answer = f"Tiningnan ko ang Labor Code document sa knowledge base, pero hindi ko nahanap ang **Article {number}**."
        elif language == "hil":
            answer = f"Gintan-aw ko ang Labor Code document sa knowledge base, pero wala ko nakita ang **Article {number}**."
        else:
            answer = f"I checked the Labor Code document in the knowledge base, but I could not find **Article {number}**."
        return {
            "answer": answer,
            "sources": [],
            "intent": "labor_code_reference_not_found",
            "kb_found": False,
        }

    if chapter_match and not extract_requested_chapter(query):
        chapter_value = chapter_match.group("chapter").upper()
        suggestion = ""
        if chapter_value.isdigit() and extract_requested_article(f"article {chapter_value}"):
            suggestion = f" If you meant **Article {chapter_value}**, ask: **article {chapter_value}**."
        if language == "fil":
            answer = f"Tiningnan ko ang Labor Code document sa knowledge base, pero hindi ko nahanap ang **Chapter {chapter_value}**.{suggestion}"
        elif language == "hil":
            answer = f"Gintan-aw ko ang Labor Code document sa knowledge base, pero wala ko nakita ang **Chapter {chapter_value}**.{suggestion}"
        else:
            answer = f"I checked the Labor Code document in the knowledge base, but I could not find **Chapter {chapter_value}**.{suggestion}"
        return {
            "answer": answer,
            "sources": [],
            "intent": "labor_code_reference_not_found",
            "kb_found": False,
        }

    return None


def answer_chapter_article_list(query: str, language: str = "en") -> Optional[Dict[str, Any]]:
    if not is_chapter_article_list_query(query):
        return None

    index = load_labor_code_index()
    chapter = extract_requested_chapter(query)
    if not chapter:
        return None

    article_lines = [
        f"- **ART. {article['number']}.** {article['title']}."
        for article in chapter["articles"]
    ]
    article_count = len(chapter["articles"])

    if language == "fil":
        intro = (
            f"Sige. Ang hinahanap mo ay ang mga Article sa **{chapter['heading']}**."
        )
        body = f"Ang mga Article sa **{chapter['heading']}** ay:"
        count_text = f"May **{article_count}** Articles sa chapter na ito."
    elif language == "hil":
        intro = (
            f"Sige. Ang ginapangita mo amo ang mga Article sa **{chapter['heading']}**."
        )
        body = f"Ang mga Article sa **{chapter['heading']}** amo ini:"
        count_text = f"May **{article_count}** ka Articles sa sini nga chapter."
    else:
        intro = (
            f"Sure. Here are the Articles listed under **{chapter['heading']}**."
        )
        body = f"The Articles inside **{chapter['heading']}** are:"
        count_text = f"There are **{article_count}** Articles in this chapter."

    answer = "\n\n".join([
        intro,
        count_text,
        body + "\n" + "\n".join(article_lines),
        (
            "Legal Basis:\n"
            f"- Labor Code of the Philippines, {chapter['heading']}\n"
            f"- Source: {index['source']}"
        ),
    ])

    return {
        "answer": answer,
        "sources": [{
            "label": "Source 1",
            "source": index["source"],
            "document_title": index["source"],
            "source_name": "DOLE",
            "source_url": index.get("source_url", LABOR_CODE_SOURCE_URL),
            "law_version": "2022 renumbered",
            "page": "",
            "citation": chapter["heading"],
            "score": 1.0,
        }],
        "intent": "chapter_article_lookup",
        "kb_found": True,
    }
