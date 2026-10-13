"""Question-level reporting using the existing legal subject classifier."""
import re
from routers.services.legal_topic_matrix import classify_topic, GENERAL_LABOR_TOPIC, NON_LEGAL_TOPIC

LABELS = {
    "thirteenth_month_pay": "13th-Month Pay",
    "termination": "Termination and Dismissal",
    "procedural_due_process": "Dismissal Notices and Due Process",
    "domestic_workers": "Kasambahay Rights",
    "workplace_safety": "Workplace Safety Rules",
    "social_security": "SSS Contributions and Benefits",
    "philhealth": "PhilHealth Benefits",
    "pagibig": "Pag-IBIG Contributions and Benefits",
    "labor_complaint": "Filing Labor Complaints",
    "ofw": "Overseas Employment",
    "safe_spaces": "Safe Spaces and Gender-Based Harassment",
    "jurisprudence_request": "Labor Court Decisions",
    'leave_recall': 'Work During Approved Leave',
    'final_pay': 'Final Pay and Clearance',
    'greeting': 'Greetings and Small Talk',
    GENERAL_LABOR_TOPIC: "Topic Not Yet Identified",
    NON_LEGAL_TOPIC: "Non-labor Questions",
}


def question_topic(text, intent=''):
    """Reporting-only wording coverage; do not change legal retrieval policy."""
    text = re.sub(r'[-\u2010-\u2015]', ' ', str(text or '').lower())
    text = re.sub(r'\bwgae\b', 'wage', text)
    if intent in {'small_talk', 'greeting'} or re.fullmatch(r'(hello|hi|hey|thanks|thank you|salamat)([ !,.]|how are you)*', text.strip()):
        return 'greeting'
    if re.search(r'\bfinal\s+pay\b', text):
        return 'final_pay'
    if re.search(r'\bleave\b', text) and re.search(r'approved|recall', text) and re.search(r'work|boss|pagtrabahuhin|sumasagot|trabaho|recall', text):
        return 'leave_recall'
    if re.search(r'\b(rest day|scheduled rest day)\b', text) and re.search(r'work|pay|computed', text):
        return 'premium_pay'
    if re.search(r'\b(oraohra|government service|government employees|civil service)\b', text):
        return 'government_employment'
    return classify_topic(text)


def conversation_topics(history):
    """Resolve in chronological order, without borrowing from a later topic."""
    topics, sessions = {}, {}
    for row in sorted(history, key=lambda r: (str(r.get('session_id', '')), int(r.get('id') or 0))):
        state = sessions.setdefault(row.get('session_id'), {'topic': GENERAL_LABOR_TOPIC, 'assistant': None})
        if row.get('role') == 'assistant':
            state['assistant'] = row
            continue
        if row.get('role') != 'user':
            continue
        text = str(row.get('content') or '')
        topic = question_topic(text, row.get('intent'))
        previous = state['assistant'] or {}
        # An explicit new subject always wins. Generic clarification menus must
        # not replace the subject of the user's original question.
        numeric_wage_reply = (topic == 'payment_of_wages' and previous.get('intent') == 'needs_clarification'
                              and len(text.split()) <= 12 and bool(re.search(r'\d', text)))
        if topic == GENERAL_LABOR_TOPIC or numeric_wage_reply:
            is_reply = previous.get('intent') == 'needs_clarification' or (
                len(text.split()) <= 12 and not re.search(r'\b(what is|explain|tell me about|ano ang)\b', text, re.I))
            if is_reply:
                topic = state['topic']
                if topic == GENERAL_LABOR_TOPIC:
                    topic = question_topic(previous.get('content'))
        topics[row['id']] = topic
        state['topic'] = topic
        state['assistant'] = None
    return topics


def rank_question_topics(rows, history=()):
    """Count each question once; examples are limited, but counts are not."""
    groups = {}
    contextual = conversation_topics(history)
    for row in rows:
        question = str(row.get("content") or "")
        topic = contextual.get(row.get('id'), question_topic(question, row.get('intent')))
        group = groups.setdefault(topic, {
            "key": topic, "label": LABELS.get(topic, topic.replace("_", " ").title()),
            "count": 0, "examples": [],
        })
        group["count"] += int(row.get("count") or 0)
        if len(group["examples"]) < 5:
            group["examples"].append({"question": question, "count": int(row.get("count") or 0)})
    return sorted(groups.values(), key=lambda row: (-row["count"], row["label"]))
