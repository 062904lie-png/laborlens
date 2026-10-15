# Follow-up interpretation and reply language

The LLM now interprets an existing conversation before the ordinary-conversation
shortcut, and its standalone question and topic hints feed NLP and retrieval.
The decision is reused within the request. Explicit new topics discard unrelated
history; saved history remains scoped by both session ID and authenticated user.

Expanded the recent window from 20 to 50 messages and from 8,000 to 32,000
characters. Long messages keep their beginning and closing question. Generation
no longer truncates that window a second time to the first 900 characters.
This remains bounded recent context, not unlimited lifetime conversation memory.
Query interpretation has its own 16,000-token request allowance so the ranking
budget does not silently skip longer conversation interpretation.

The same conversation's saved sector, worker type, and employment status now
reach intake and answer generation as user facts. If the model still asks which
sector when one is already known, the backend drops that redundant question and
continues with the known scope. Guest chats recover these facts only from their
own supplied conversation history. A user correction updates the fact.

The same interpretation call selects English, Filipino, or Hiligaynon in Auto
mode from the user's words, including a language switch after a previous reply.
Explicit UI language selections remain authoritative. A substantive message can
also change the classifier fallback language; short ambiguous replies retain the
previous language.

Validation: 48 focused unittest cases pass, including actual mocked HTTP transport
with expanded history, clarification chains, prior user facts, explicit language
selection, Hiligaynon selection, and independent concurrent histories. Updated an
old assertion to inspect every body-search call because recovery can add a later
search. Python compilation passes.

Live synthetic interpretation checks demonstrated correct Hiligaynon follow-up
and new-topic handling, but repeated runs also encountered timeouts, inconsistent
language selection, and rejected output. These are not claimed as fully passing
live quality tests. Provider unavailability still uses contextual fallback;
neither perfect language detection nor a fixed response latency is guaranteed.
No production conversations were created or deleted for these checks.

Deployment artifacts and rollback-capable activation are in
`_lightsail_upload/followup-language-20261015`. Runtime environment, knowledge base,
database contents, model pool configuration, and frontend were preserved.
