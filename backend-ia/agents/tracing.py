"""
OpenTelemetry tracing setup for the Amael-IA agent layer.
Exports a pre-configured tracer that pushes spans to the OTel Collector
already running in the observability namespace (otel-collector:4317).

Usage:
    from agents.tracing import tracer
    with tracer.start_as_current_span("my.span") as span:
        span.set_attribute("key", "value")
        ...
"""
import os
import logging

logger = logging.getLogger(__name__)

OTEL_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://otel-collector.observability.svc.cluster.local:4317",
)
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "amael-backend")

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

    tracer = trace.get_tracer(SERVICE_NAME)
    _otel_available = True
    logger.info(f"[TRACING] OpenTelemetry configurado → {OTEL_ENDPOINT}")

except ImportError:
    # OTEL packages not installed — provide a no-op tracer so the rest of the
    # code does not need to handle the missing dependency.
    import contextlib

    class _NoOpSpan:
        def set_attribute(self, *a, **kw): pass
        def record_exception(self, *a, **kw): pass
        def set_status(self, *a, **kw): pass

    class _NoOpTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name, **kw):
            yield _NoOpSpan()

    tracer = _NoOpTracer()
    _otel_available = False
    logger.warning("[TRACING] opentelemetry-sdk no instalado. Usando tracer no-op.")


def instrument_app(app):
    """Call once at FastAPI startup to auto-instrument HTTP requests."""
    if _otel_available:
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        logger.info("[TRACING] FastAPI y HTTPX instrumentados con OpenTelemetry.")
