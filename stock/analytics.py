"""5년 주가 분석, 밸류에이션 지표, 차트 데이터 생성."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

from stock.config import StockContext
from stock.pipeline import build_analysis_summary

logger = logging.getLogger("predict_stock")

VALUATION_KEYS = {
    "trailingPE": "PER (후행)",
    "forwardPE": "PER (선행)",
    "priceToBook": "PBR",
    "priceToSalesTrailing12Months": "PSR",
    "enterpriseToEbitda": "EV/EBITDA",
    "pegRatio": "PEG",
    "beta": "베타",
    "dividendYield": "배당수익률",
    "marketCap": "시가총액",
    "trailingEps": "EPS (후행)",
    "bookValue": "BPS",
    "fiftyTwoWeekHigh": "52주 최고",
    "fiftyTwoWeekLow": "52주 최저",
}


def fetch_valuation_metrics(ticker: str) -> dict:
    """Yahoo Finance에서 PER, PBR 등 밸류에이션 지표를 가져옵니다."""
    result = {"available": False, "metrics": {}, "raw": {}}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("밸류에이션 조회 실패 (%s): %s", ticker, exc)
        return result

    metrics = {}
    for key, label in VALUATION_KEYS.items():
        value = info.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if key == "dividendYield" and isinstance(value, (int, float)):
            value = float(value) * 100
        if key == "marketCap" and isinstance(value, (int, float)):
            value = float(value)
        metrics[label] = value
        result["raw"][key] = value

    result["metrics"] = metrics
    result["available"] = len(metrics) > 0
    result["currency"] = info.get("currency", "")
    result["sector"] = info.get("sector", "")
    result["industry"] = info.get("industry", "")
    return result


def _series_to_list(series: pd.Series) -> list:
    return [None if pd.isna(v) else float(v) for v in series]


def build_chart_data(df: pd.DataFrame) -> dict:
    """웹 차트용 시계열·분포 데이터를 생성합니다."""
    work = df.copy().sort_values("Date").reset_index(drop=True)
    dates = [d.strftime("%Y-%m-%d") for d in work["Date"]]
    close = work["Close"]
    volume = work["Volume"]

    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    daily_ret = close.pct_change().dropna() * 100
    rolling_vol = close.pct_change().rolling(30, min_periods=10).std() * np.sqrt(252) * 100

    # 일간 수익률 히스토그램 (-10% ~ +10%, 40 bins)
    hist_counts, hist_edges = np.histogram(
        daily_ret.clip(-10, 10), bins=40, range=(-10, 10)
    )
    hist_labels = [
        f"{hist_edges[i]:.1f}~{hist_edges[i + 1]:.1f}" for i in range(len(hist_counts))
    ]

    # 월별 수익률
    monthly = work.set_index("Date")["Close"].resample("ME").last().pct_change().dropna() * 100

    # 변동성 지표
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) else 0.0

    return {
        "price_trend": {
            "dates": dates,
            "close": _series_to_list(close),
            "ma20": _series_to_list(ma20),
            "ma60": _series_to_list(ma60),
        },
        "volume": {
            "dates": dates,
            "values": [float(v) for v in volume],
        },
        "daily_returns_hist": {
            "labels": hist_labels,
            "counts": [int(c) for c in hist_counts],
        },
        "rolling_volatility": {
            "dates": dates,
            "values": _series_to_list(rolling_vol),
        },
        "monthly_returns": {
            "labels": [d.strftime("%Y-%m") for d in monthly.index],
            "values": [float(v) for v in monthly.values],
        },
        "risk_metrics": {
            "annualized_volatility_pct": ann_vol,
            "max_drawdown_pct": _calc_max_drawdown(close),
            "sharpe_ratio": _calc_sharpe(daily_ret),
        },
    }


def _calc_max_drawdown(close: pd.Series) -> float:
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max * 100
    return float(drawdown.min()) if len(drawdown) else 0.0


def _calc_sharpe(daily_ret: pd.Series, risk_free: float = 0.0) -> float:
    if daily_ret.empty or daily_ret.std() == 0:
        return 0.0
    excess = daily_ret.mean() * 252 - risk_free
    vol = daily_ret.std() * np.sqrt(252)
    return float(excess / vol)


def build_full_analysis(
    stock: StockContext,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: LinearRegression,
    metrics: dict,
) -> dict:
    """통계 요약 + 밸류에이션 + 차트 데이터를 통합합니다."""
    summary = build_analysis_summary(stock, df, train_df, test_df, model, metrics)
    valuation = fetch_valuation_metrics(stock.ticker)
    charts = build_chart_data(df)

    summary["valuation"] = valuation
    summary["charts"] = charts
    summary["risk_metrics"] = charts.get("risk_metrics", {})
    return summary


def save_analysis_files(stock: StockContext, analysis: dict) -> None:
    """analysis.json 및 charts.json 저장."""
    import json
    from pathlib import Path

    stock.ensure_dirs()
    analysis_path = stock.model_dir / "analysis.json"
    charts_path = stock.model_dir / "charts.json"

    charts = analysis.pop("charts", {})
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    with open(charts_path, "w", encoding="utf-8") as f:
        json.dump(charts, f, ensure_ascii=False, indent=2)
    analysis["charts"] = charts


def load_charts_data(stock: StockContext) -> dict | None:
    """저장된 차트 데이터를 로드하거나 DataFrame에서 생성합니다."""
    import json
    from pathlib import Path

    from stock.pipeline import load_csv_as_ohlcv, _legacy_paths_if_needed

    charts_path = stock.model_dir / "charts.json"
    if charts_path.exists():
        with open(charts_path, encoding="utf-8") as f:
            return json.load(f)

    legacy = _legacy_paths_if_needed(stock)
    if legacy["cleaned_file"].exists():
        df = load_csv_as_ohlcv(legacy["cleaned_file"])
        return build_chart_data(df)
    return None


def ensure_full_analysis(stock: StockContext) -> dict | None:
    """분석 데이터가 없으면 CSV·모델에서 재생성합니다."""
    from stock.pipeline import _build_analysis_from_files, load_csv_as_ohlcv, _legacy_paths_if_needed

    analysis_path = stock.model_dir / "analysis.json"
    if analysis_path.exists():
        import json
        with open(analysis_path, encoding="utf-8") as f:
            analysis = json.load(f)
        charts = load_charts_data(stock)
        if charts:
            analysis["charts"] = charts
            if not analysis.get("risk_metrics"):
                analysis["risk_metrics"] = charts.get("risk_metrics", {})
        if not analysis.get("valuation"):
            analysis["valuation"] = fetch_valuation_metrics(stock.ticker)
        return analysis

    rebuilt = _build_analysis_from_files(stock)
    if not rebuilt:
        return None

    legacy = _legacy_paths_if_needed(stock)
    if legacy["cleaned_file"].exists():
        df = load_csv_as_ohlcv(legacy["cleaned_file"])
        rebuilt["charts"] = build_chart_data(df)
        rebuilt["valuation"] = fetch_valuation_metrics(stock.ticker)
        rebuilt["risk_metrics"] = rebuilt["charts"].get("risk_metrics", {})
        save_analysis_files(stock, dict(rebuilt))
        rebuilt["charts"] = load_charts_data(stock) or rebuilt.get("charts")
    return rebuilt
