"""Use saved source IDs as discovery hints, never historical prose as authority."""
from .source_history import parse_saved_sources


def previous_sources(query, resolved_query, history, *, relation=None):
    if relation == 'new_topic' or (relation != 'follow_up'
            and str(query or '').strip() == str(resolved_query or '').strip()):
        return []
    for message in reversed(list(history or [])[-20:]):
        if message.get('role') != 'assistant':
            continue
        if message.get('intent') == 'needs_clarification' or message.get('clarification_question'):
            continue
        if str(message.get('source_match') or '').lower().replace(' ', '_') in {
                'insufficient', 'conflicting', 'not_required', 'provider_unavailable', 'unverified'}:
            return []
        sources = parse_saved_sources(message.get('sources'))
        # Do not jump across an unrelated source-free answer to an older topic.
        if not sources:
            return []
        return [s for s in sources if str(s.get('doc_db_id') or '').isdigit()][:30]
    return []


def previous_source_ids(query, resolved_query, history):
    return list(dict.fromkeys(str(s['chunk_id']) for s in previous_sources(query, resolved_query, history)
                             if s.get('chunk_id')))


async def rehydrate_followup_sources(query, resolved_query, history, *, relation=None):
    """Read authoritative bodies afresh; callers must supply owner-scoped history."""
    sources = previous_sources(query, resolved_query, history, relation=relation)
    if not sources:
        return []
    import asyncio
    from . import chroma_service as store
    ids = list(dict.fromkeys(str(s['chunk_id']) for s in sources if s.get('chunk_id')))
    operations = []
    if ids:
        operations.append(asyncio.to_thread(store.get_collection().get, ids=ids, include=['documents', 'metadatas']))
    # Older source cards did not save chunk IDs. Read only their exact document
    # and page, bounded to four lookups, rather than trusting saved excerpts or
    # scanning a whole document. These are discovery hints, not legal authority.
    pages = []
    for source in sources:
        if source.get('chunk_id'):
            continue
        try:
            page = int(source.get('page'))
        except (TypeError, ValueError):
            continue
        key = (str(source['doc_db_id']), page)
        if page > 0 and key not in pages:
            pages.append(key)
    for doc_id, page in pages[:4]:
        operations.append(asyncio.to_thread(store.get_collection().get,
            where={'$and': [{'doc_db_id': doc_id}, {'$or': [
                {'page_number': page}, {'page_number': str(page)},
                {'page': page}, {'page': str(page)}]}]},
            limit=12, include=['documents', 'metadatas']))
    chunks = []
    for records in await asyncio.gather(*operations):
        chunks.extend(store._records_from_get_results(records, score=0.0,
            retrieval_method='conversation_source_rehydration', enrich_metadata=False))
    # Always validate the registry and exact active index version, including
    # when IDs still exist after a reindex, retirement, or staged failed repair.
    from routers.chat import filter_searchable_chunks
    return [c for c in await filter_searchable_chunks(chunks, resolved_query)
            if str((c.get('metadata') or {}).get('doc_db_id') or '').isdigit()]
