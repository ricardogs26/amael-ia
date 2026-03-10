"""
OpenTelemetry tracing setup for productivity-service.
Mirrors the pattern from backend-ia/agents/tracing.py.
"""
import os
import logging

logger = logging.getLogger(__name__)

OTEL_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://otel-collector.observability.svc.cluster.local:4317",
)
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "amael-productivity")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    _resource = Resource.create({"service.name": SERVICE_NAME})
    _provider = TracerProvider(resource=_resource)
    _exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
    _provider.add_span_processor(BatchSpanProcessor(_exporter))
    trace.set_tracer_provider(_provider)

    # Auto-instrument outbound httpx calls (Google API calls, Vault)
    HTTPXClientInstrumentor().instrument()

    _otel_available = True
    logger.info(f"[TRACING] OpenTelemetry configurado → {OTEL_ENDPOINT} (service={SERVICE_NAME})")

except ImportError as e:
    _otel_available = False
    logger.warning(f"[TRACING] opentelemetry no instalado ({e}). Usando no-op.")


def instrument_app(app):
    """Call once after FastAPI app creation to auto-instrument HTTP requests."""
    if _otel_available:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[TRACING] FastAPI instrumentado con OpenTelemetry.")
