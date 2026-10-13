"""Reconstruct bounded clarification chains from user facts, across topics."""
import re


def _asks_for_missing_detail(message):
    """Recognize an inline clarification without treating assistant prose as facts."""
    if message.get('intent') == 'needs_clarification' or message.get('clarification_question'):
        return True
    text = str(message.get('content') or '')
    # Partial answers can finish with a model-written question in any supported
    # language. Retain only the ensuing USER reply as a fact, not this prose.
    tail = [part.strip() for part in text.strip().split('\n\n')[-2:]]
    if any(part.endswith('?') and part.count('?') == 1 and len(part) <= 500 for part in tail):
        return True
    return bool(re.search(
        r'\b(?:please (?:specify|provide|clarify|confirm)|'
        r'(?:which|what|how many)\b[^?.!]{0,160}\?|'
        r'(?:not specified|not provided|missing)\b[^.!?]{0,160}'
        r'(?:require|need|specify|detail)|'
        r'(?:require|need)\w*\b[^.!?]{0,100}(?:that detail|this detail|to know))',
        text, re.I))


_CONTEXTUAL_EXAMPLE_REQUEST_RE = re.compile(
    r"^\s*(?:(?:please|pls)\s+)?"
    r"(?:(?:can|could|would)\s+you\s+(?:(?:please)\s+)?)?"
    r"(?:give|show|provide|work\s+out|do)\s+(?:(?:me|us)\s+)?"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:(?:worked|simple|numerical)\s+)?example|sample)"
    r"(?:\s+(?:calculation|computation))?\s*[?.!]*\s*$",
    re.I,
)


def _resolve_contextual_example_request(query, messages):
    """Attach an elliptical example request to the immediately answered topic.

    Only the user's prior question and any short user replies are carried over;
    assistant prose remains context, never search evidence or user fact.
    """
    example = _CONTEXTUAL_EXAMPLE_REQUEST_RE.fullmatch(str(query or ''))
    contextual = re.match(r'^\s*(?:what\s+if|what\s+about|how\s+about|how\s+much\s+(?:for|if)|'
                          r'and\s+(?:for|if)|using\s+(?:the\s+)?same\s+(?:rate|formula))\b',
                          str(query or ''), re.I)
    if not example and not contextual:
        return None
    if len(str(query or '')) > 400 or re.search(r'\b(?:new|different)\s+(?:question|topic|scenario)\b', str(query or ''), re.I):
        return None
    assistant_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if messages[index].get('role') == 'assistant'),
        None,
    )
    if assistant_index is None:
        return None
    assistant = messages[assistant_index]
    if assistant.get('intent') == 'needs_clarification' or assistant.get('clarification_question'):
        return None

    replies = []
    for message in reversed(messages[:assistant_index]):
        if message.get('role') != 'user':
            continue
        content = str(message.get('content') or '').strip()
        if not content:
            continue
        from .issue_evidence import issue_terms
        # A bare named subject can be the original request, not a missing fact.
        position = next((i for i, m in enumerate(messages) if m is message), -1)
        previous_prompt = messages[position - 1] if position > 0 else {}
        if is_short_reply(content) and (_asks_for_missing_detail(previous_prompt)
                                       or (re.search(r'\d', content) and len(content.split()) <= 8)):
            replies.append(content)
            continue
        if not issue_terms(content) or re.fullmatch(r'(?:hi|hello|hey|thanks|thank you|kamusta)[.! ]*', content, re.I):
            return None
        context = content[:1200]
        if replies:
            context += '\nUser-provided details: ' + '; '.join(reversed(replies))
        return ('Subject context (earlier user question, not an additional current request): '
                + context + '\nCurrent follow-up question: ' + str(query or '').strip())
    return None


def is_short_reply(text):
    text = str(text or '').strip()
    return bool(text and len(text) <= 240 and len(text.split()) <= 30
                and '?' not in text
                and not re.match(r'^(?:how|what|why|can|is|are|does|do|ano|paano|bakit|new question|different question|change (?:the )?topic|ibang tanong)\b', text, re.I))


def resolve_followup_context(query, history):
    messages = list(history or [])
    if messages and messages[-1].get('role') == 'user' and str(messages[-1].get('content', '')).strip() == query.strip():
        messages.pop()
    contextual_example = _resolve_contextual_example_request(query, messages)
    if contextual_example:
        return contextual_example
    # A reply to a clarification can contain several sentences. Do not discard
    # the earlier scenario just because the user gave a complete explanation.
    # Still leave a clearly new question on its own, even if a clarification
    # happened in the previous turn.
    if not is_short_reply(query):
        latest_assistant = next((m for m in reversed(messages) if m.get('role') == 'assistant'), {})
        pending_clarification = _asks_for_missing_detail(latest_assistant)
        clearly_new_question = bool(re.match(
            r'^(?:how|what|why|can|is|are|does|do|should|will|when|where|who|'
            r'ano|paano|bakit|saan|sino|new question|different question|change (?:the )?topic|ibang tanong)\b',
            str(query or '').strip(), re.I,
        ))
        if not pending_clarification or clearly_new_question:
            return query
    # Find the nearest substantive user question without crossing another topic.
    replies = [query.strip()]
    prompts = []
    for message in reversed(messages[-20:]):
        content = str(message.get('content') or '').strip()
        if message.get('role') == 'assistant':
            if _asks_for_missing_detail(message):
                prompts.append(len(replies))
            continue
        if message.get('role') != 'user':
            continue
        if prompts:
            # The user turn preceding a clarification may itself be an answer
            # to an earlier clarification. Keep walking only if that is explicit.
            index = next((i for i, m in enumerate(messages) if m is message), -1)
            prior = messages[index - 1] if index > 0 else {}
            if is_short_reply(content) and _asks_for_missing_detail(prior):
                replies.append(content)
                continue
            # Responses given after the most recent clarification are corrections;
            # use the latest one, not conflicting older choices.
            facts = [query.strip()]
            if len(prompts) > 1:
                facts = list(reversed(replies[prompts[0]:])) + facts
            return content[:1200] + '\nUser clarification (latest answer takes precedence): ' + '; '.join(facts)
        if not is_short_reply(content):
            return query
        replies.append(content)
    return query
