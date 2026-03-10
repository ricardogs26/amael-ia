# app/services/auth_service.py
import os
import json
import logging

import hvac
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Vault configuration ───────────────────────────────────────────────────────
# The pod's K8s service account JWT is injected by the kubelet automatically.
_K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_VAULT_MOUNT = "secret"
_VAULT_BASE_PATH = "amael/google-tokens"  # KV v2: secret/data/amael/google-tokens/{key}


def _sanitize(email: str) -> str:
    """Convert email to a safe Vault path segment."""
    return email.replace("@", "_at_").replace(".", "_dot_")


def _vault_client() -> hvac.Client:
    """
    Authenticate to Vault using the pod's Kubernetes service account JWT.
    Requires VAULT_ADDR and VAULT_ROLE env vars.
    """
    vault_addr = settings.vault_addr
    vault_role = settings.vault_role

    client = hvac.Client(url=vault_addr)
    with open(_K8S_TOKEN_PATH) as f:
        jwt = f.read().strip()

    client.auth.kubernetes.login(role=vault_role, jwt=jwt)
    if not client.is_authenticated():
        raise RuntimeError(f"Vault auth failed (addr={vault_addr}, role={vault_role})")
    return client


# ── Public API ────────────────────────────────────────────────────────────────
def get_auth_flow() -> Flow:
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        redirect_uri=settings.redirect_uri,
    )


def get_user_credentials(user_email: str) -> Credentials | None:
    """
    Retrieve Google OAuth credentials from Vault KV v2.
    Stored at: secret/data/amael/google-tokens/{sanitized_email}
    """
    try:
        client = _vault_client()
        path = f"{_VAULT_BASE_PATH}/{_sanitize(user_email)}"
        response = client.secrets.kv.v2.read_secret_version(
            path=path, mount_point=_VAULT_MOUNT
        )
        creds_data = response["data"]["data"]
        creds = Credentials.from_authorized_user_info(creds_data)
        logger.info(f"[VAULT] Credenciales cargadas para {user_email}")
        return creds
    except hvac.exceptions.InvalidPath:
        logger.info(f"[VAULT] Sin credenciales para {user_email} (primera vez)")
        return None
    except Exception as exc:
        logger.warning(f"[VAULT] Error obteniendo credenciales de {user_email}: {exc}")
        return None


def save_user_credentials(user_email: str, creds: Credentials) -> None:
    """
    Store Google OAuth credentials in Vault KV v2 (per-user, encrypted at rest).
    Path: secret/data/amael/google-tokens/{sanitized_email}
    """
    try:
        client = _vault_client()
        creds_dict = json.loads(creds.to_json())
        path = f"{_VAULT_BASE_PATH}/{_sanitize(user_email)}"
        client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=creds_dict,
            mount_point=_VAULT_MOUNT,
        )
        logger.info(f"[VAULT] Credenciales guardadas para {user_email}")
    except Exception as exc:
        logger.error(f"[VAULT] Error guardando credenciales de {user_email}: {exc}")
        raise