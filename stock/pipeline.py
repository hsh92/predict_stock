"""데이터 수집, 전처리, 학습, 예측 파이프라인."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from stock.config import StockContext

logger = logging.getLogger("predict_stock")

TRAIN_RATIO = 0.8
FEATURE_COLUMNS = ["Open", "High", "Low", "Volume"]
TARGET_COLUMN = "Close"
STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 또는 CSV 데이터를 표준 OHLCV 형식으로 변환합니다."""
    work = df.copy()

    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [str(col[0]) for col in work.columns]

    if "Date" not in work.columns:
        if work.index.name in ("Date", "Datetime") or isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index()
            if "Datetime" in work.columns:
                work = work.rename(columns={"Datetime": "Date"})
            elif work.columns[0] != "Date":
                work = work.rename(columns={work.columns[0]: "Date"})

    rename_map = {}
    for col in work.columns:
        lower = str(col).lower()
        if lower in ("open", "high", "low", "close", "volume", "date"):
            rename_map[col] = lower.capitalize() if lower != "date" else "Date"
    work = work.rename(columns=rename_map)

    missing = [col for col in STANDARD_COLUMNS if col not in work.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    work = work[STANDARD_COLUMNS].copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna().sort_values("Date").reset_index(drop=True)
    if work.empty:
        raise ValueError("유효한 주가 데이터가 없습니다.")
    return work


def load_csv_as_ohlcv(path) -> pd.DataFrame:
    """저장된 CSV(yfinance 원본 또는 정규화본)를 로드합니다."""
    raw = pd.read_csv(path)
    first_col = str(raw.columns[0]).lower()

    if first_col == "price" or "ticker" in str(raw.iloc[0].values).lower():
        df = pd.read_csv(path, skiprows=[0, 1])
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return normalize_ohlcv_dataframe(df)

    return normalize_ohlcv_dataframe(raw)


def download_stock_data(stock: StockContext, years: int = 5) -> pd.DataFrame:
    """Yahoo Finance에서 최근 N년 일봉 데이터를 다운로드합니다."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    logger.info(
        "Yahoo Finance 데이터 수집: %s (%s), %s ~ %s",
        stock.ticker,
        stock.name,
        start_date.date(),
        end_date.date(),
    )

    df = yf.download(
        stock.ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"'{stock.ticker}' 데이터를 가져오지 못했습니다. 티커를 확인하세요."
        )

    normalized = normalize_ohlcv_dataframe(df)
    stock.ensure_dirs()
    normalized.to_csv(stock.raw_file, index=False, encoding="utf-8-sig")
    logger.info("원본 데이터 저장: %s (%s행)", stock.raw_file, len(normalized))
    return normalized


def preprocess_data(df: pd.DataFrame, stock: StockContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """NaN 제거 후 훈련/테스트 분할 및 CSV 저장."""
    df = df.dropna().reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    stock.ensure_dirs()
    train_df.to_csv(stock.train_file, index=False, encoding="utf-8-sig")
    test_df.to_csv(stock.test_file, index=False, encoding="utf-8-sig")
    df.to_csv(stock.cleaned_file, index=False, encoding="utf-8-sig")
    return train_df, test_df, df


def train_model(train_df: pd.DataFrame) -> tuple[LinearRegression, StandardScaler]:
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_model(model, scaler, test_df: pd.DataFrame) -> dict:
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values
    y_pred = model.predict(scaler.transform(X_test))
    return {
        "r2_score": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "test_samples": len(test_df),
    }


def build_analysis_summary(
    stock: StockContext,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: LinearRegression,
    metrics: dict,
) -> dict:
    """모델 학습에 사용된 데이터·모델 분석 요약을 생성합니다."""
    close = df["Close"]
    daily_change_pct = close.pct_change().dropna() * 100
    volume = df["Volume"]

    coefficients = {}
    if hasattr(model, "coef_") and len(model.coef_) == len(FEATURE_COLUMNS):
        for feature, coef in zip(FEATURE_COLUMNS, model.coef_):
            coefficients[feature] = float(coef)

    return {
        "ticker": stock.ticker,
        "name": stock.name,
        "exchange": stock.exchange,
        "data_period": {
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end": df["Date"].max().strftime("%Y-%m-%d"),
            "total_days": int(len(df)),
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "train_ratio": TRAIN_RATIO,
        },
        "price_stats": {
            "close_mean": float(close.mean()),
            "close_std": float(close.std()),
            "close_min": float(close.min()),
            "close_max": float(close.max()),
            "close_latest": float(close.iloc[-1]),
        },
        "volume_stats": {
            "volume_mean": float(volume.mean()),
            "volume_max": float(volume.max()),
        },
        "daily_change_stats": {
            "mean_pct": float(daily_change_pct.mean()),
            "std_pct": float(daily_change_pct.std()),
            "max_up_pct": float(daily_change_pct.max()),
            "max_down_pct": float(daily_change_pct.min()),
        },
        "model_performance": {
            "r2_score": metrics["r2_score"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "intercept": float(model.intercept_) if hasattr(model, "intercept_") else 0.0,
            "coefficients": coefficients,
        },
        "saved_at": datetime.now().isoformat(),
    }


def save_model(
    stock: StockContext,
    model,
    scaler,
    metrics: dict,
    df: pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict:
    stock.ensure_dirs()
    with open(stock.model_file, "wb") as f:
        pickle.dump(model, f)
    with open(stock.scaler_file, "wb") as f:
        pickle.dump(scaler, f)

    analysis = {}
    if df is not None and train_df is not None and test_df is not None:
        from stock.analytics import build_full_analysis, save_analysis_files

        analysis = build_full_analysis(stock, df, train_df, test_df, model, metrics)
        save_analysis_files(stock, dict(analysis))

    model_info = {
        **metrics,
        "ticker": stock.ticker,
        "name": stock.name,
        "exchange": stock.exchange,
        "saved_at": datetime.now().isoformat(),
        "analysis": analysis,
    }
    with open(stock.model_info_file, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    if analysis:
        analysis_file = stock.model_dir / "analysis.json"
        # charts는 save_analysis_files에서 저장됨; analysis에 charts 키 복원
        from stock.analytics import load_charts_data
        charts = load_charts_data(stock)
        if charts:
            analysis["charts"] = charts

    return analysis


def load_analysis_data(stock: StockContext) -> dict | None:
    """저장된 분석 데이터를 로드합니다."""
    from stock.analytics import ensure_full_analysis

    return ensure_full_analysis(stock)


def _build_analysis_from_files(stock: StockContext) -> dict | None:
    """분석 JSON이 없을 때 CSV·모델 파일에서 요약을 생성합니다."""
    legacy = _legacy_paths_if_needed(stock)
    cleaned_file = legacy["cleaned_file"]
    model_file = legacy["model_file"]

    if not cleaned_file.exists() or not model_file.exists():
        return None

    try:
        df = load_csv_as_ohlcv(cleaned_file)
        split_idx = int(len(df) * TRAIN_RATIO)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        info_file = stock.model_info_file
        if stock.ticker == "005930.KS" and not info_file.exists():
            info_file = Path("models/model_info.json")

        metrics = {"r2_score": 0, "rmse": 0, "mae": 0, "test_samples": len(test_df)}
        if info_file.exists():
            with open(info_file, encoding="utf-8") as f:
                info = json.load(f)
            metrics = {
                "r2_score": info.get("r2_score", 0),
                "rmse": info.get("rmse", 0),
                "mae": info.get("mae", 0),
                "test_samples": info.get("test_samples", len(test_df)),
            }

        return build_analysis_summary(stock, df, train_df, test_df, model, metrics)
    except Exception as exc:
        logger.warning("분석 데이터 생성 실패 (%s): %s", stock.ticker, exc)
        return None


def run_full_pipeline(
    stock: StockContext,
    status_callback: Callable[[str, str, int], None] | None = None,
    fetch_from_yahoo: bool = True,
) -> dict:
    """수집 → 전처리 → 학습 → 평가 → 저장 전체 파이프라인."""

    def notify(step: str, message: str, progress: int) -> None:
        if status_callback:
            status_callback(step, message, progress)

    notify("데이터 수집", f"{stock.name} ({stock.ticker}) 데이터 다운로드 중...", 10)

    if fetch_from_yahoo:
        df = download_stock_data(stock)
    elif stock.raw_file.exists():
        df = load_csv_as_ohlcv(stock.raw_file)
    elif stock.cleaned_file.exists():
        df = load_csv_as_ohlcv(stock.cleaned_file)
    else:
        raise FileNotFoundError(f"{stock.ticker} 데이터 파일이 없습니다.")

    notify("데이터 수집", f"[OK] 완료: {len(df)}행 데이터 수집됨", 20)

    notify("데이터 전처리", "NaN 제거 및 훈련/테스트 분할 중...", 30)
    train_df, test_df, cleaned_df = preprocess_data(df, stock)
    notify(
        "데이터 전처리",
        f"[OK] 완료: 훈련({len(train_df)}) / 테스트({len(test_df)})",
        40,
    )

    notify("모델 학습", "선형 회귀 모델 학습 중...", 55)
    model, scaler = train_model(train_df)
    notify("모델 학습", "[OK] 완료: 모델 학습됨", 65)

    notify("모델 평가", "테스트 데이터로 모델 평가 중...", 75)
    metrics = evaluate_model(model, scaler, test_df)
    notify(
        "모델 평가",
        f"[OK] 완료: R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.2f}",
        85,
    )

    notify("모델 저장", "모델과 정보 저장 중...", 90)
    save_model(stock, model, scaler, metrics, cleaned_df, train_df, test_df)
    notify("모델 저장", "[OK] 완료: 모델 저장됨", 95)

    return metrics


def load_model_and_data(stock: StockContext):
    """종목별 모델과 정제 데이터를 로드합니다."""
    legacy = _legacy_paths_if_needed(stock)
    model_file = legacy["model_file"]
    scaler_file = legacy["scaler_file"]
    cleaned_file = legacy["cleaned_file"]

    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        df = load_csv_as_ohlcv(cleaned_file)
        return model, scaler, df
    except FileNotFoundError as exc:
        logger.warning("모델/데이터 로드 실패 (%s): %s", stock.ticker, exc)
        return None, None, None


def _legacy_paths_if_needed(stock: StockContext) -> dict:
    """삼성전자 레거시 단일 경로 하위 호환."""
    if stock.ticker == "005930.KS":
        legacy_model = Path("models/linear_regression_model.pkl")
        legacy_cleaned = Path("data/processed/samsung_005930_cleaned.csv")
        if legacy_model.exists() and not stock.model_file.exists():
            return {
                "model_file": legacy_model,
                "scaler_file": Path("models/feature_scaler.pkl"),
                "cleaned_file": legacy_cleaned,
            }
    return {
        "model_file": stock.model_file,
        "scaler_file": stock.scaler_file,
        "cleaned_file": stock.cleaned_file,
    }


def predict_stock(model, scaler, df: pd.DataFrame, stock: StockContext | None = None) -> dict:
    """내일 종가를 예측합니다."""
    latest_row = df.iloc[-1]
    features = np.array(
        [[latest_row["Open"], latest_row["High"], latest_row["Low"], latest_row["Volume"]]]
    )
    predicted_close = float(model.predict(scaler.transform(features))[0])
    today_close = float(latest_row["Close"])

    result = {
        "today_date": latest_row["Date"].strftime("%Y-%m-%d"),
        "tomorrow_date": (latest_row["Date"] + timedelta(days=1)).strftime("%Y-%m-%d"),
        "today_open": float(latest_row["Open"]),
        "today_high": float(latest_row["High"]),
        "today_low": float(latest_row["Low"]),
        "today_close": today_close,
        "today_volume": int(latest_row["Volume"]),
        "predicted_close": predicted_close,
        "expected_change": predicted_close - today_close,
        "expected_change_pct": (predicted_close - today_close) / today_close * 100,
    }

    if stock:
        result["ticker"] = stock.ticker
        result["stock_name"] = stock.name
        result["exchange"] = stock.exchange

    return result
