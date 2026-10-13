"""Bounded cross-category candidate discovery; never a substitute for reranking."""
import asyncio

from routers.services.kb_categories import KB_CATEGORIES


async def search_category_coverage(
    search,
    build_filter,
    query,
    profile,
    top_k=6,
    *,
    include_amendments=True,
    global_queries=None,
):
    """Search every category, then uncategorized records and update provisions.

    Separate category budgets prevent a populous category from crowding out
    relevant jurisprudence or guidance. The caller applies eligibility, scope,
    deduplication and evidence checks to all returned candidates.
    """
    semaphore = asyncio.Semaphore(3)

    async def run(category):
        async with semaphore:
            records = await search(
                query=query, n_results=top_k,
                filter_metadata=build_filter([category], profile, match_type="any"),
            )
        return records, {"stage": "category_coverage", "categories": [category],
                         "retrieved": len(records), "top_k": top_k}

    results = await asyncio.gather(*(run(category) for category in KB_CATEGORIES))
    chunks = [chunk for records, _ in results for chunk in records]
    stages = [stage for _, stage in results]
    # No category filter: covers missing, custom and legacy category metadata.
    probes = [("global_coverage", query)]
    for index, fallback_query in enumerate(global_queries or [], start=1):
        value = str(fallback_query or "").strip()
        if value and value not in {probe[1] for probe in probes}:
            probes.append((f"global_user_query_{index}", value))
    if include_amendments:
        probes.append(("amendment_coverage", query + " | amendments amended revised implementing rules superseding effectivity repeal"))
    for stage, search_query in probes:
        records = await search(query=search_query, n_results=12, filter_metadata=None)
        chunks.extend(records)
        stages.append({"stage": stage, "retrieved": len(records), "top_k": 12})
    return chunks, stages
