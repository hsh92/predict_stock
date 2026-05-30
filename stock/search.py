"""Yahoo Finance 종목 검색."""

from __future__ import annotations

import logging
import re

import requests
import yfinance as yf

logger = logging.getLogger("predict_stock")

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
USER_AGENT = "Mozilla/5.0 (compatible; PredictStock/1.0)"


def _normalize_query(query: str) -> list[str]:
    """검색어 변형 목록 (한국 6자리 종목코드 등)."""
    query = query.strip()
    variants = [query]
    if re.fullmatch(r"\d{6}", query):
        variants.append(f"{query}.KS")
        variants.append(f"{query}.KQ")
    return variants


def _parse_quote(item: dict) -> dict | None:
    symbol = item.get("symbol")
    if not symbol or "-USD" in symbol or symbol.endswith("=F"):
        return None

    quote_type = item.get("quoteType")
    if quote_type not in (None, "EQUITY", "ETF"):
        return None

    name = item.get("longname") or item.get("shortname") or symbol
    exchange = item.get("exchange") or item.get("exchDisp") or ""

    return {
        "ticker": symbol,
        "name": name,
        "exchange": exchange,
        "quote_type": quote_type or "EQUITY",
        "sector": item.get("sectorDisp") or item.get("sector") or "",
    }


def _search_yfinance(query: str, max_results: int) -> list[dict]:
    result = yf.Search(query, max_results=max_results)
    quotes = result.quotes or []
    stocks: list[dict] = []
    seen: set[str] = set()

    for item in quotes:
        parsed = _parse_quote(item)
        if not parsed or parsed["ticker"] in seen:
            continue
        seen.add(parsed["ticker"])
        stocks.append(parsed)

    return stocks


def _search_yahoo_api(query: str, max_results: int) -> list[dict]:
    response = requests.get(
        YAHOO_SEARCH_URL,
        params={"q": query, "quotesCount": max_results, "lang": "ko-KR"},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    quotes = response.json().get("quotes", [])

    stocks: list[dict] = []
    seen: set[str] = set()
    for item in quotes:
        parsed = _parse_quote(item)
        if not parsed or parsed["ticker"] in seen:
            continue
        seen.add(parsed["ticker"])
        stocks.append(parsed)
    return stocks


def search_stocks(query: str, max_results: int = 10) -> list[dict]:
    """Yahoo Finance에서 종목명/티커/종목코드로 검색합니다."""
    query = (query or "").strip()
    if len(query) < 1:
        return []

    all_results: list[dict] = []
    seen: set[str] = set()

    try:
        for variant in _normalize_query(query):
            for searcher in (_search_yfinance, _search_yahoo_api):
                try:
                    batch = searcher(variant, max_results)
                except Exception as exc:
                    logger.debug("검색 실패 (%s, %s): %s", searcher.__name__, variant, exc)
                    continue

                for item in batch:
                    if item["ticker"] not in seen:
                        seen.add(item["ticker"])
                        all_results.append(item)

            if all_results:
                break

    except Exception as exc:
        logger.error("종목 검색 실패: %s", exc, exc_info=True)
        raise RuntimeError(f"종목 검색에 실패했습니다: {exc}") from exc

    return all_results[:max_results]
