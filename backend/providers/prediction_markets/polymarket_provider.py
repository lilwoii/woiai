import os
from typing import Any, Dict, List, Optional

class PolymarketProvider:
    """Thin wrapper around Polymarket's official py-clob-client.

    This file is designed to import cleanly even if py-clob-client isn't installed.
    """

    def __init__(self):
        self.host = os.getenv("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
        self.chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))  # Polygon mainnet is commonly used
        self.funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self.api_key = os.getenv("POLYMARKET_API_KEY", "")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET", "")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "")

        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            # Official client package
            # pip install py-clob-client
            from py_clob_client.client import ClobClient
        except Exception as e:
            raise RuntimeError(
                "py-clob-client is not installed. Install with: pip install py-clob-client"
            ) from e

        # The py-clob-client supports public methods without auth.
        # Auth headers are required for trading / L2 methods.
        self._client = ClobClient(
            host=self.host,
            chain_id=self.chain_id,
            private_key=self.private_key or None,
            api_key=self.api_key or None,
            api_secret=self.api_secret or None,
            api_passphrase=self.api_passphrase or None,
            funder=self.funder or None,
        )

    # ------------- Public data -------------

    def search_markets(self, q: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a list of markets matching query. Uses CLOB metadata endpoints where possible."""
        self._ensure_client()
        # py-clob-client does not provide a single universal 'search' method across all versions.
        # We'll use the public 'get_markets' endpoint if available, otherwise raise.
        client = self._client
        if hasattr(client, "get_markets"):
            data = client.get_markets()
            markets = data.get("markets", data) if isinstance(data, dict) else data
        else:
            raise RuntimeError("py-clob-client version missing get_markets(). Update the package.")

        ql = (q or "").strip().lower()
        out = []
        for m in markets:
            title = (m.get("question") or m.get("title") or "")
            if not ql or ql in title.lower():
                out.append({
                    "id": m.get("condition_id") or m.get("id") or m.get("market_id"),
                    "title": title,
                    "active": m.get("active", True),
                    "raw": m,
                })
            if len(out) >= limit:
                break
        return out

    def get_orderbook(self, token_id: str) -> Dict[str, Any]:
        self._ensure_client()
        client = self._client
        if not hasattr(client, "get_order_book"):
            raise RuntimeError("py-clob-client missing get_order_book().")
        return client.get_order_book(token_id)

    def get_mid_price(self, token_id: str) -> Optional[float]:
        ob = self.get_orderbook(token_id)
        bids = ob.get("bids", []) or []
        asks = ob.get("asks", []) or []
        try:
            best_bid = float(bids[0]["price"]) if bids else None
            best_ask = float(asks[0]["price"]) if asks else None
            if best_bid is None or best_ask is None:
                return best_bid or best_ask
            return (best_bid + best_ask) / 2.0
        except Exception:
            return None

    # ------------- Trading (requires auth) -------------

    def place_limit_order(self, token_id: str, side: str, price: float, size: float) -> Dict[str, Any]:
        self._ensure_client()
        client = self._client
        if not hasattr(client, "create_order"):
            raise RuntimeError("py-clob-client missing create_order().")
        # create_order typically returns signed order payload; then post_order submits.
        order = client.create_order(
            token_id=token_id,
            price=float(price),
            size=float(size),
            side=side.lower(),
        )
        if hasattr(client, "post_order"):
            return client.post_order(order)
        if hasattr(client, "create_and_post_order"):
            return client.create_and_post_order(order)
        raise RuntimeError("py-clob-client missing post_order().")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        self._ensure_client()
        client = self._client
        if hasattr(client, "cancel_order"):
            return client.cancel_order(order_id)
        raise RuntimeError("py-clob-client missing cancel_order().")
