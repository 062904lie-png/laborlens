"""No-op Chroma product telemetry client.

Chroma 0.5.x can call an incompatible three-argument ``posthog.capture``
signature even when anonymized telemetry is disabled. Retrieval must not be
affected by that optional integration, so use an explicit no-op client.
"""

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent


class DisabledProductTelemetry(ProductTelemetryClient):
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None

    # ChromaDB's bundled ``overrides`` metaclass checks this marker while the
    # class is created. Setting it directly avoids a version-specific decorator
    # mismatch between ChromaDB and the installed overrides package.
    capture.__override__ = True
