import os
from typing import Any, Dict, List, Optional

class KalshiProvider:
    """Wrapper around the official Kalshi Python SDK.

    Kalshi's docs currently recommend kalshi_python_sync / kalshi_python_async.
    We'll support sync SDK usage behind an async-friendly wrapper.
    """

    def __init__(self):
        self.key_id = os.getenv("KALSHI_API_KEY_ID", "")
        self.private_key_pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
        self.env = os.getenv("KALSHI_ENV", "prod")  # prod or demo
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import kalshi_python
            from kalshi_python import Configuration, ApiClient
            from kalshi_python.api_client import ApiClient as ApiClient2
        except Exception as e:
            raise RuntimeError(
                "Kalshi SDK is not installed. Install with: pip install kalshi_python_sync"
            ) from e

        from kalshi_python import Configuration, KalshiClient

        cfg = Configuration()
        # The SDK uses api_key_id + private_key_pem for auth
        cfg.api_key_id = self.key_id or None
        cfg.private_key_pem = self.private_key_pem or None

        # Environment selection differs across SDK versions; keep default if not present.
        if hasattr(cfg, "host") and self.env == "demo":
            # Some versions provide demo host; if not, user can override via env.
            cfg.host = os.getenv("KALSHI_HOST", cfg.host)

        self._client = KalshiClient(cfg)

    def search_markets(self, q: str, limit: int = 20) -> List[Dict[str, Any]]:
        self._ensure_client()
        client = self._client

        # SDK structure: client.markets_api.get_markets(...) in many versions.
        if not hasattr(client, "markets_api"):
            raise RuntimeError("Kalshi SDK missing markets_api.")
        api = client.markets_api

        # We'll call get_markets with a keyword filter if supported; otherwise fetch and filter.
        markets = None
        if hasattr(api, "get_markets"):
            try:
                markets = api.get_markets(limit=limit, search=q)
            except TypeError:
                markets = api.get_markets(limit=limit)
        else:
            raise RuntimeError("Kalshi SDK missing get_markets().")

        # Response shape varies (dict-like or model). Normalize.
        items = getattr(markets, "markets", None) or getattr(markets, "items", None) or markets
        out = []
        ql = (q or "").lower().strip()
        for m in (items or []):
            title = getattr(m, "title", None) or (m.get("title") if isinstance(m, dict) else "")
            mid = getattr(m, "ticker", None) or (m.get("ticker") if isinstance(m, dict) else None)
            if not ql or ql in (title or "").lower():
                out.append({"id": mid, "title": title, "raw": m})
            if len(out) >= limit:
                break
        return out

    # Trading endpoints are intentionally left minimal here; they require user setup & credentials.
