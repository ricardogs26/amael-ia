"""
P4 — Input validation and output sanitization for the chat pipeline.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ── Input limits ──────────────────────────────────────────────────────────────
MAX_PROMPT_CHARS = 4_000

# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"
    r"|forget\s+(everything|all|your|the)",
    re.IGNORECASE,
)

# ── Output redaction patterns ─────────────────────────────────────────────────
# Vault service tokens (hvs. prefix)
_VAULT_TOKEN_RE = re.compile(r"\bhvs\.[A-Za-z0-9]{20,}\b")
# JWT tokens (three base64url segments)
_JWT_RE = re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
# Generic high-entropy secrets (long alphanumeric+special strings after common key= patterns)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(password|secret|token|api_key|apikey)\s*[:=]\s*['\"]?([A-Za-z0-9+/=!@#$%^&*]{16,})['\"]?",
)


def validate_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validates the user prompt before it enters the agent pipeline.
    Returns (is_valid, sanitized_prompt_or_error_message).
    """
    # Strip null bytes and non-printable control characters (keep newlines/tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)

    if len(cleaned) > MAX_PROMPT_CHARS:
        logger.warning(f"[SECURITY] Prompt demasiado largo: {len(cleaned)} chars (max {MAX_PROMPT_CHARS}).")
        return False, f"El mensaje es demasiado largo ({len(cleaned)} caracteres). El máximo permitido es {MAX_PROMPT_CHARS}."

    if _INJECTION_PATTERNS.search(cleaned):
        logger.warning(f"[SECURITY] Patrón de prompt injection detectado: {cleaned[:100]!r}")
        return False, "El mensaje contiene patrones no permitidos."

    return True, cleaned


def sanitize_output(text: str) -> str:
    """
    Redacts secrets that should never appear in user-facing responses.
    """
    if not text:
        return text

    original_len = len(text)

    text = _VAULT_TOKEN_RE.sub("[VAULT_TOKEN_REDACTED]", text)
    text = _JWT_RE.sub("[JWT_REDACTED]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group(1)}=[REDACTED]", text
    )

    if len(text) != original_len:
        logger.warning("[SECURITY] Se redactaron posibles secretos de la respuesta.")

    return text
