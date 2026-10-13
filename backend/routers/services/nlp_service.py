"""
LaborLens – NLP Service
Uses trained multilingual classifiers and learned entity extraction.
"""

import spacy
import re
from typing import Dict, List, Any
import logging
from datetime import date
from . import learned_nlp

logger = logging.getLogger(__name__)
_nlp = None

SUPPORTED_LANGUAGES = {"en", "fil", "hil"}


LEGAL_INTENTS = {
    "wage_inquiry",
    "leave_benefits",
    "termination_rights",
    "13th_month",
    "employment_status",
    "occupational_safety",
    "holiday_pay",
    "general_inquiry",
}

ORDINARY_CONVERSATION_INTENTS = {
    "greeting",
    "small_talk",
    "identity",
    "capability_question",
    "gratitude",
    "farewell",
}


def detect_legal_intent(text: str) -> str:
    intent = detect_intent(text)
    return intent if intent in LEGAL_INTENTS else ""


def normalize_labor_terms(text: str) -> str:
    """Normalize whitespace without guessing translations or correcting meaning.

    Keep this public helper for existing callers. Learned classification and
    entity extraction still receive the original question.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_topic_hints(hints) -> List[str]:
    """Normalize a small set of model-proposed search phrases, not legal facts."""
    result, seen = [], set()
    if not isinstance(hints, (list, tuple)):
        return result
    for hint in list(hints)[:4]:
        if not isinstance(hint, str):
            continue
        normalized = re.sub(r"\s+", " ", normalize_labor_terms(hint)).strip(" \t.,:—–-")
        if not 2 <= len(normalized) <= 120 or any(char in normalized for char in '<>\n\r?!;'):
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def load_nlp_model():
    global _nlp
    try:
        _nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy model loaded.")
    except OSError as exc:
        raise RuntimeError('Install en_core_web_sm during setup; NLP does not download models while serving requests.') from exc


def detect_language(text: str) -> str:
    return learned_nlp.detect_language(text)


def detect_intent(text: str) -> str:
    return learned_nlp.classify(text)["intent"]


def model_configuration():
    return {'spacy_version': spacy.__version__, 'model': 'learned-multilingual-classifier',
            'model_version': learned_nlp.metadata()['model_version'],
            'search_normalization': 'whitespace_only',
            'manual_search_expansion': False,
            'auxiliary_spacy_model': 'en_core_web_sm',
            'auxiliary_spacy_version': (_nlp.meta or {}).get('version') if _nlp else None,
            'learned_classifier': learned_nlp.metadata()}


def extract_entities(text: str) -> Dict[str, List[str]]:
    if _nlp is None:
        load_nlp_model()
    doc      = _nlp(text)
    iso_dates = []
    for match in re.finditer(r'(?<![\w-])\d{4}-\d{2}-\d{2}(?![\w-])', text):
        try:
            date.fromisoformat(match.group())
        except ValueError:
            continue
        iso_dates.append((match.start(), match.end(), match.group()))
    entities = {"money": [], "dates": [], "organizations": [], "law_references": [], "numbers": []}
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["money"].append(ent.text)
        elif ent.label_ in ("DATE", "TIME"):
            if not any(ent.start_char < end and ent.end_char > start for start, end, _ in iso_dates):
                entities["dates"].append(ent.text)
        elif ent.label_ in ("CARDINAL", "ORDINAL"):
            entities["numbers"].append(ent.text)
    entities['dates'].extend(value for _, _, value in iso_dates)
    # spaCy's NER often drops Philippine peso amounts, especially when the
    # symbol and comma-grouped digits are adjacent (for example, ₱15,000).
    # Recover explicit currency values deterministically so personalized
    # calculations do not ask users to repeat amounts already in their query.
    amount_pattern = re.compile(
        r"(?<![\w])(?:₱|PHP|P)\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?!\d)|"
        r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?\s*(?:PHP|pesos?)(?!\w)",
        re.IGNORECASE,
    )
    for match in amount_pattern.finditer(text):
        amount = match.group(0).strip()
        if amount.casefold() not in {value.casefold() for value in entities["money"]}:
            entities["money"].append(amount)
    law_refs = re.findall(
        r"\b(?:article\s+\d+|R\.?\s*A\.?\s*\d+|P\.?\s*D\.?\s*\d+|republic act\s+\d+|presidential decree\s+\d+|book\s+[IVX]+)\b",
        text, re.IGNORECASE
    )
    entities["law_references"] = law_refs
    entities['organizations'] = list(learned_nlp.organizations(text))
    for kind, values in entities.items():
        # A model and a rule can return the same span with different casing.
        entities[kind] = list({value.casefold(): value for value in values}.values())
    return entities


def preprocess_query(text: str, topic_hints=None) -> Dict[str, Any]:
    if _nlp is None:
        load_nlp_model()
    normalized = normalize_labor_terms(text)
    classification = dict(learned_nlp.classify(text))
    classification['routing_override'] = classification['intent'] != classification['candidate']
    classification['decision_source'] = (
        'learned_content_route' if classification['routing_override'] else 'learned_intent'
    )
    language = classification['language']
    intent = classification['intent']
    legal_intent = intent if intent in LEGAL_INTENTS else ''
    entities = extract_entities(text)
    doc      = _nlp(normalized)
    # A stop-word bag loses negation, quantities and relationships. Retain the
    # original wording for search so negation and named entities remain intact.
    clean = normalized
    search_text = clean or normalized
    noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 2]
    return {
        "original":     text,
        "normalized":   normalized,
        "language":     language,
        "intent":       intent,
        "conversational_phrases": [],
        "legal_intent": legal_intent,
        "classification": classification,
        "entities":     entities,
        "noun_phrases": noun_phrases,
        "clean_text":   clean or normalized,
        "search_text":  search_text,
        # Kept separate from the question and its extracted facts. Retrieval
        # may use these as soft search hints; they never gate documents or
        # become evidence for answer generation.
        "search_hints": normalize_topic_hints(topic_hints),
    }


load_nlp_model()
