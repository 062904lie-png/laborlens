"""Sector-first routing layered over topic categories."""
from routers.services.kb_categories import routed_kb_categories

CIVIL_SERVICE_CATEGORY = "Civil Service / Government Employment"


def sector_category_route(topic, sector):
    if sector == "government_employment":
        return [CIVIL_SERVICE_CATEGORY], [CIVIL_SERVICE_CATEGORY, "Jurisprudence"]
    return (routed_kb_categories(topic, limit=3),
            routed_kb_categories(topic, expanded=True, limit=50))
