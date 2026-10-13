"""Bounded, read-only parent retrieval from the same physical document version."""
import asyncio
import hashlib
import json
import re

from config import settings
from .legal_topic_matrix import source_passage_text
from .request_budget import estimate_request_tokens


def _identity(meta):
    owner, version = meta.get('doc_db_id'), meta.get('index_version')
    return (str(owner), str(version)) if owner not in (None, '') and version not in (None, '') else None


def _parent(meta):
    role = meta.get('chunk_role')
    if role == 'document_section_parent' and meta.get('section_key'):
        return ('section_key', str(meta['section_key']), 'section_part_index', 'section_part_count')
    if role in {'document_article_parent', 'article_parent'} and meta.get('article_number'):
        return ('article_number', str(meta['article_number']), 'article_chunk_index', 'article_chunk_count')
    return None


def _assemble(seed, parts):
    """Only a complete, unique sequence can be presented as a complete section."""
    meta = seed.get('metadata') or {}
    parent = _parent(meta)
    if not parent or not _identity(meta):
        return None
    field, value, index_field, count_field = parent
    try:
        count = int(meta[count_field])
        if not 1 <= count <= int(getattr(settings, 'LABORLENS_SECTION_CONTEXT_MAX_PARTS', 32)):
            return None
        matching = [p for p in parts if _identity(p.get('metadata') or {}) == _identity(meta)
                    and (p.get('metadata') or {}).get('chunk_role') == meta.get('chunk_role')
                    and str((p.get('metadata') or {}).get(field)) == value]
        # Duplicates, missing parts or contradictory counts fail closed. The
        # anchor's occurrence/hierarchy prevents combining repeated sections.
        if len(matching) != count or any(int(p['metadata'][count_field]) != count for p in matching):
            return None
        if any(str(p['metadata'].get('is_active', True)).lower() in {'false', '0', 'no'}
               or str(p['metadata'].get('legal_status') or p['metadata'].get('version_status') or '').lower()
               not in {'', 'active', 'current', 'approved', 'indexed'} for p in matching):
            return None
        matching.sort(key=lambda p: int(p['metadata'][index_field]))
        if [int(p['metadata'][index_field]) for p in matching] != list(range(count)):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    bodies = [source_passage_text(p.get('text') or '').strip() for p in matching]
    if not all(bodies):
        return None
    # Parent parts are packed non-overlapping ranges at ingestion. Do not guess
    # overlap from a matching letter or combine unrelated ordinary chunks.
    body = '\n\n'.join(bodies)
    if len(body) > int(getattr(settings, 'LABORLENS_SECTION_CONTEXT_CHAR_LIMIT', 3500)):
        return None
    if estimate_request_tokens([{'content': body}], 0) > int(getattr(settings, 'LABORLENS_SECTION_CONTEXT_TOKEN_LIMIT', 1200)):
        return None
    ids = [str(p.get('id') or '') for p in matching]
    fingerprint = hashlib.sha256(json.dumps([_identity(meta), ids, body]).encode()).hexdigest()
    try:
        pages = [int(p['metadata'].get('page') or 0) for p in matching]
        ends = [int(p['metadata'].get('page_end') or p['metadata'].get('page') or 0) for p in matching]
    except (TypeError, ValueError):
        return None
    if any(page < 0 or end < page for page, end in zip(pages, ends)):
        return None
    return {**seed, 'id': 'section-context:' + fingerprint,
            'text': body, 'metadata': {**matching[0]['metadata'],
                'chunk_id': 'section-context:' + fingerprint,
                'chunk_role': 'section_context', 'page': min(pages), 'page_end': max(ends),
                'context_member_ids': json.dumps(ids), 'section_context_complete': True,
                'retrieval_method': 'complete_section'}}


def _continuations(meta, parts):
    """Keep whole nearby parts of an oversized parent, never guess topic neighbors."""
    parent, identity = _parent(meta), _identity(meta)
    if not parent or not identity:
        return []
    field, value, index_field, count_field = parent
    try:
        anchor, count = int(meta[index_field]), int(meta[count_field])
        if not 0 <= anchor < count <= int(getattr(settings, 'LABORLENS_SECTION_CONTEXT_MAX_PARTS', 32)):
            return []
        siblings = [p for p in parts if _identity(p.get('metadata') or {}) == identity
                    and p['metadata'].get('chunk_role') == meta.get('chunk_role')
                    and str(p['metadata'].get(field)) == value]
        indexes = [int(p['metadata'][index_field]) for p in siblings]
        if len(indexes) != len(set(indexes)) or any(
                not 0 <= index < count for index in indexes):
            return []
        if any(int(p['metadata'][count_field]) != count
               or str(p['metadata'].get('is_active', True)).lower() in {'false', '0', 'no'}
               or str(p['metadata'].get('legal_status') or p['metadata'].get('version_status') or '').lower()
               not in {'', 'active', 'current', 'approved', 'indexed'} for p in siblings):
            return []
        return [{**p, 'metadata': {**p['metadata'], 'section_context_complete': False,
                                  'retrieval_method': 'section_continuation'}}
                for p in sorted(siblings, key=lambda p: int(p['metadata'][index_field]))
                if abs(int(p['metadata'][index_field]) - anchor) == 1
                and source_passage_text(p.get('text') or '').strip()]
    except (KeyError, TypeError, ValueError):
        return []


async def expand_complete_sections(seeds, limit=5, *, token_budget=None):
    """Expand parent sections, reserving capacity for explicit references.

    No provider/embedding calls or writes. Older chunks without usable parent
    metadata retain their original excerpts. Oversized or incomplete parents
    are not silently clipped and labelled complete.
    """
    from . import chroma_service as chroma
    if chroma._collection is None or not seeds:
        return seeds
    max_parts = max(1, min(64, int(getattr(settings, 'LABORLENS_SECTION_CONTEXT_MAX_PARTS', 32))))
    expansion_limit = 8 if token_budget is not None else 2

    async def read(meta, extra, *, operation_stage='section_part_reconstruction_read'):
        identity = _identity(meta)
        if not identity:
            return []
        fields = [{'doc_db_id': identity[0]}, {'index_version': identity[1]}, *extra]
        from .performance import current_retrieval_diagnostics
        diagnostics = current_retrieval_diagnostics()
        operation = lambda: asyncio.to_thread(chroma._collection.get,
            where={'$and': fields}, limit=max_parts + 1, include=['documents', 'metadatas'])
        records = await (diagnostics.run(operation_stage, operation, candidate_count=lambda value: len(value.get('ids') or []))
                         if diagnostics else operation())
        return [p for p in chroma._records_from_get_results(records, score=0,
                retrieval_method='section_part', enrich_metadata=False)
                if _identity(p.get('metadata') or {}) == identity]

    async def expand():
        output, references, processed, covered_ids = [], {}, set(), set()
        for seed in seeds:
            if seed.get('id') in covered_ids:
                continue  # Already included verbatim in a complete parent.
            meta = seed.get('metadata') or {}
            parent = _parent(meta)
            if not parent and _identity(meta) and meta.get('page') not in (None, '') and len(processed) < expansion_limit:
                # Locate a parent by a literal body overlap, not a title/topic
                # assertion. Check several interior windows for split anchors.
                parents = await read(meta, [
                    {'chunk_role': {'$in': ['document_section_parent', 'document_article_parent']}},
                    {'page': {'$lte': int(meta['page'])}},
                    {'page_end': {'$gte': int(meta['page'])}}],
                    operation_stage='parent_location_reconstruction_read')
                body = source_passage_text(seed.get('text') or '')
                windows = [body[i:i + 100] for i in (0, len(body) // 2, max(0, len(body) - 100))]
                matches = [p for p in parents if any(len(w.strip()) >= 80 and w in source_passage_text(p['text']) for w in windows)]
                keys = {(_parent(p['metadata']) or (None, None))[:2] for p in matches}
                if len(keys) == 1 and matches:
                    meta = matches[0]['metadata']
                    parent = _parent(meta)
            key = (_identity(meta), parent[:2] if parent else None)
            if not parent or len(processed) >= expansion_limit or key in processed:
                output.append(seed)
                continue
            processed.add(key)
            parts = await read(meta, [{'chunk_role': meta['chunk_role']}, {parent[0]: parent[1]}],
                               operation_stage='section_part_reconstruction_read')
            from .performance import current_retrieval_diagnostics
            diagnostics = current_retrieval_diagnostics()
            context = (diagnostics.measure_sync('section_parent_assembly',
                        lambda: _assemble({**seed, 'metadata': meta}, parts),
                        input_count=len(parts), candidate_count=lambda value: int(bool(value)))
                       if diagnostics else _assemble({**seed, 'metadata': meta}, parts))
            output.append(context or seed)
            if context:
                covered_ids.update(json.loads(context['metadata']['context_member_ids']))
            elif token_budget is not None:
                # A long/incomplete parent is not a complete section. Carry
                # adjacent same-section parts as a bounded evidence bundle,
                # even when their categories or LDA topics differ.
                references.setdefault(seed['id'], []).extend(_continuations(meta, parts))
            if not context or meta.get('chunk_role') != 'document_section_parent':
                continue
            # One level, same-version explicit local references only. Do not
            # resolve bare numbers or guess a target in another law or hierarchy.
            targets = list(dict.fromkeys(re.findall(
                r'\bSection\s+(\d+[A-Z]?)\s+of\s+this\s+(?:Rule|Chapter|Act)\b', context['text'], re.I)))[:(8 if token_budget is not None else 1)]
            for number in targets:
                records = await read(meta, [{'chunk_role': 'document_section_parent'},
                                            {'section_number': number.upper()}],
                                    operation_stage='referenced_section_expansion_read')
                records = [p for p in records if p['metadata'].get('section_path', '') == meta.get('section_path', '')]
                if len({p['metadata'].get('section_key') for p in records}) != 1:
                    continue
                ref = _assemble(records[0], records) if records else None
                if ref and ref['id'] != context['id']:
                    references.setdefault(context['id'], []).append(ref)
        # Pack each rule and its discovered qualifications as a unit. Appending
        # references after all seeds silently discarded them at a full limit.
        # A higher-ranked rule's dependency takes precedence over a later seed;
        # never retain that expanded rule alone when its bundle cannot fit.
        result, retained_ids, retained_members = [], set(), set()
        for passage in output:
            if passage.get('id') in retained_ids | retained_members:
                continue
            bundle, bundle_ids = [], set()
            for item in [passage, *references.get(passage.get('id'), [])]:
                identity = item.get('id')
                if identity not in retained_ids | retained_members | bundle_ids:
                    bundle.append(item)
                    bundle_ids.add(identity)
            if limit is not None and len(result) + len(bundle) > max(0, limit):
                continue
            if token_budget is not None:
                from .evidence_selection import evidence_tokens
                if evidence_tokens(result + bundle) > token_budget:
                    continue
                if len(bundle) > 1:
                    bundle = [{**item, 'metadata': {**(item.get('metadata') or {}),
                        'evidence_bundle': str(passage['id'])}} for item in bundle]
            result.extend(bundle)
            retained_ids.update(bundle_ids)
            for item in bundle:
                retained_members.update(json.loads((item.get('metadata') or {}).get('context_member_ids') or '[]'))
        return seeds if result == seeds else result
    try:
        return await asyncio.wait_for(expand(), timeout=2)
    except Exception:
        return seeds  # Original evidence remains usable; verification still applies.
