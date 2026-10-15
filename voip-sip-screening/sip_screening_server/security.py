"""Webhook signature verification per provider.

Twilio: X-Twilio-Signature header, base64(HMAC-SHA1(auth_token, full_url +
sorted POST params)). Documented by Twilio and third-party references.
Plivo:  X-Plivo-Signature-V2 header plus X-Plivo-Signature-V2-Nonce,
        HMAC-SHA256(auth_token, base_url + nonce), per Plivo docs.
Exotel: TBD. The applet URL-resolution pattern uses HTTP Basic auth; the
        status-callback signature scheme must be confirmed against current
        Exotel docs before production use. verify_exotel raises until then.
"""

from __future__ import annotations

import base64
import hashlib
import hmac


def _hmac_b64(key: str, message: str, digest: str) -> str:
    mac = hmac.new(key.encode("utf-8"), message.encode("utf-8"),
                   getattr(hashlib, digest))
    return base64.b64encode(mac.digest()).decode("ascii")


def verify_twilio_signature(url: str, params: dict[str, str],
                            signature: str, auth_token: str) -> bool:
    """Verify a Twilio form-encoded webhook.

    url: the exact full URL Twilio called (scheme + host + path, no port
    juggling surprises; honor X-Forwarded-Proto/Host when behind a proxy).
    params: the POST form parameters as sent.
    """
    if not signature or not auth_token:
        return False
    data = url + "".join(k + params[k] for k in sorted(params))
    expected = _hmac_b64(auth_token, data, "sha1")
    return hmac.compare_digest(expected, signature)


def verify_plivo_signature(base_url: str, nonce: str, signature: str,
                           auth_token: str) -> bool:
    """Verify a Plivo webhook using the V2 signature scheme.

    base_url: the webhook URL without query string, exactly as configured.
    nonce: value of the X-Plivo-Signature-V2-Nonce header.
    """
    if not signature or not nonce or not auth_token:
        return False
    expected = _hmac_b64(auth_token, base_url + nonce, "sha256")
    return hmac.compare_digest(expected, signature)


def verify_exotel_request(*args, **kwargs) -> bool:  # noqa: ANN002, ANN003
    """Exotel webhook verification: TBD.

    Confirm the current scheme against Exotel's docs before enabling this
    provider in production. Until then, fail closed.
    """
    raise NotImplementedError(
        "Exotel webhook signature scheme is TBD; confirm against current "
        "Exotel docs before production use."
    )
