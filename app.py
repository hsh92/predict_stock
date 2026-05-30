#!/usr/bin/env python3
"""
범용 주식 예측 Flask 웹 애플리케이션

Yahoo Finance 종목 검색 → 5년 데이터 수집 → 모델 학습 → 예측
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import threading
import json

try:
    import pandas as pd
    from flask import Flask, render_template, jsonify, request
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    sys.exit(1)

from stock.config import StockContext, get_current_stock, set_current_stock
from stock.analytics import ensure_full_analysis, load_charts_data
from stock.pipeline import load_analysis_data, load_model_and_data, predict_stock, run_full_pipeline
from stock.search import search_stocks

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"


def _configure_stdio_utf8():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    app_logger = logging.getLogger("predict_stock")
    if app_logger.handlers:
        return app_logger

    app_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    app_logger.addHandler(file_handler)
    app_logger.addHandler(console_handler)

    for logger_name in ("werkzeug", "flask.app"):
        ext_logger = logging.getLogger(logger_name)
        if not any(isinstance(h, RotatingFileHandler) for h in ext_logger.handlers):
            ext_logger.addHandler(file_handler)
        ext_logger.setLevel(logging.INFO)

    return app_logger


_configure_stdio_utf8()
logger = setup_logging()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.secret_key = "predict-stock-dev-key"

workflow_status = {
    "running": False,
    "progress": 0,
    "current_step": "",
    "message": "",
    "error": None,
    "completed": False,
    "steps_completed": [],
    "workflow_type": "",
    "ticker": "",
    "stock_name": "",
    "model_r2": None,
    "model_rmse": None,
}


@app.template_filter("strftime")
def strftime_filter(date_obj, format_str):
    if date_obj is None:
        return ""
    return date_obj.strftime(format_str)


@app.template_filter("datetimeformat")
def datetimeformat(value, format="%Y-%m-%d %H:%M:%S"):
    if value is None:
        return ""
    return value.strftime(format)


@app.template_filter("format_price")
def format_price(value):
    if value is None:
        return "0"
    return f"{int(value):,}"


def update_status(step, message, progress):
    global workflow_status
    workflow_status["current_step"] = step
    workflow_status["message"] = message
    workflow_status["progress"] = progress
    if step and step not in workflow_status["steps_completed"]:
        workflow_status["steps_completed"].append(step)
    logger.info("[%s%%] %s: %s", progress, step, message)


def _run_pipeline_workflow(stock: StockContext, workflow_type: str, fetch_from_yahoo: bool):
    global workflow_status

    try:
        workflow_status = {
            "running": True,
            "progress": 0,
            "current_step": "",
            "message": "",
            "error": None,
            "completed": False,
            "steps_completed": [],
            "workflow_type": workflow_type,
            "ticker": stock.ticker,
            "stock_name": stock.name,
            "model_r2": None,
            "model_rmse": None,
        }

        set_current_stock(stock)
        logger.info("%s 워크플로우 시작: %s (%s)", workflow_type, stock.name, stock.ticker)

        metrics = run_full_pipeline(
            stock,
            status_callback=update_status,
            fetch_from_yahoo=fetch_from_yahoo,
        )

        workflow_status["progress"] = 100
        workflow_status["current_step"] = "완료"
        workflow_status["message"] = "[OK] 모든 작업이 완료되었습니다!"
        workflow_status["completed"] = True
        workflow_status["running"] = False
        workflow_status["model_r2"] = metrics["r2_score"]
        workflow_status["model_rmse"] = metrics["rmse"]
        logger.info("워크플로우 완료: %s (%s)", stock.name, stock.ticker)

    except Exception as e:
        logger.error("워크플로우 오류 (%s): %s", stock.ticker, e, exc_info=True)
        workflow_status["error"] = str(e)
        workflow_status["running"] = False


def _get_stock_or_none() -> StockContext | None:
    return get_current_stock()


def _build_dashboard_context(stock: StockContext):
    model, scaler, df = load_model_and_data(stock)
    if model is None:
        return None

    prediction = predict_stock(model, scaler, df, stock)
    recent_data = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "open": f"{row['Open']:,.0f}",
            "high": f"{row['High']:,.0f}",
            "low": f"{row['Low']:,.0f}",
            "close": f"{row['Close']:,.0f}",
            "volume": f"{row['Volume']:,.0f}",
        }
        for row in df.tail(10).to_dict("records")
    ]

    model_info = {}
    info_file = stock.model_info_file
    if info_file.exists():
        with open(info_file, encoding="utf-8") as f:
            model_info = json.load(f)

    analysis = load_analysis_data(stock) or ensure_full_analysis(stock)
    chart_data = None
    if analysis:
        chart_data = analysis.get("charts") or load_charts_data(stock)
        if chart_data and not analysis.get("charts"):
            analysis["charts"] = chart_data

    return {
        "stock": stock,
        "prediction": prediction,
        "recent_data": recent_data,
        "model_info": model_info,
        "analysis": analysis,
        "chart_data": chart_data,
        "now": datetime.now(),
    }


@app.route("/")
def index():
    stock = _get_stock_or_none()
    dashboard = _build_dashboard_context(stock) if stock else None
    has_model = dashboard is not None
    return render_template(
        "index.html",
        stock=stock if has_model else None,
        dashboard=dashboard,
        has_model=has_model,
        now=datetime.now(),
    )


@app.route("/api/analysis")
def get_analysis():
    """5년 분석 데이터 및 차트 JSON."""
    stock = _get_stock_or_none()
    if not stock:
        return jsonify({"error": "선택된 종목이 없습니다."}), 404

    analysis = ensure_full_analysis(stock)
    if not analysis:
        return jsonify({"error": "분석 데이터를 생성할 수 없습니다."}), 404

    return jsonify(analysis)


@app.route("/api/stocks/search")
def api_search_stocks():
    query = request.args.get("q", "").strip()
    if len(query) < 1:
        return jsonify({"results": []})
    try:
        results = search_stocks(query, max_results=15)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stocks/current")
def api_current_stock():
    stock = _get_stock_or_none()
    if not stock:
        return jsonify({"stock": None})
    return jsonify({"stock": stock.to_dict()})


@app.route("/api/stocks/select", methods=["POST"])
def api_select_stock():
    global workflow_status

    if workflow_status["running"]:
        return jsonify({"error": "이미 작업이 진행 중입니다."}), 400

    data = request.get_json(silent=True) or {}
    ticker = (data.get("ticker") or "").strip()
    name = (data.get("name") or ticker).strip()
    exchange = (data.get("exchange") or "").strip()

    if not ticker:
        return jsonify({"error": "티커를 선택해 주세요."}), 400

    stock = StockContext(ticker=ticker, name=name, exchange=exchange)
    logger.info("종목 선택: %s (%s)", name, ticker)

    thread = threading.Thread(
        target=_run_pipeline_workflow,
        args=(stock, "select", True),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "stock": stock.to_dict()})


@app.route("/api/prediction")
def get_prediction():
    stock = _get_stock_or_none()
    if not stock:
        return jsonify({"error": "선택된 종목이 없습니다."}), 404

    model, scaler, df = load_model_and_data(stock)
    if model is None:
        return jsonify({"error": "모델을 찾을 수 없습니다."}), 404

    return jsonify(predict_stock(model, scaler, df, stock))


@app.route("/api/history")
def get_history():
    stock = _get_stock_or_none()
    if not stock:
        return jsonify({"error": "선택된 종목이 없습니다."}), 404

    try:
        from stock.pipeline import load_csv_as_ohlcv

        cleaned = stock.cleaned_file
        if stock.ticker == "005930.KS" and not cleaned.exists():
            cleaned = Path("data/processed/samsung_005930_cleaned.csv")

        df = load_csv_as_ohlcv(cleaned)
        days = request.args.get("days", 30, type=int)
        history = df.tail(days)[["Date", "Close"]]
        return jsonify(
            {
                "dates": [d.strftime("%Y-%m-%d") for d in history["Date"]],
                "closes": history["Close"].tolist(),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workflow/status")
def get_workflow_status():
    return jsonify(workflow_status)


@app.route("/api/retrain/status")
def get_retrain_status():
    return jsonify(workflow_status)


@app.route("/api/retrain/start", methods=["POST"])
def start_retrain():
    global workflow_status

    if workflow_status["running"]:
        return jsonify({"error": "이미 작업이 진행 중입니다."}), 400

    stock = _get_stock_or_none()
    if not stock:
        return jsonify({"error": "먼저 종목을 검색하여 선택해 주세요."}), 400

    logger.info("재학습 요청: %s (%s)", stock.name, stock.ticker)
    thread = threading.Thread(
        target=_run_pipeline_workflow,
        args=(stock, "retrain", True),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "retrain started"})


@app.route("/api/retrain/reset", methods=["POST"])
def reset_retrain():
    global workflow_status
    if workflow_status["running"]:
        return jsonify({"error": "작업 진행 중에는 초기화할 수 없습니다."}), 400

    workflow_status = {
        "running": False,
        "progress": 0,
        "current_step": "",
        "message": "",
        "error": None,
        "completed": False,
        "steps_completed": [],
        "workflow_type": "",
        "ticker": "",
        "stock_name": "",
        "model_r2": None,
        "model_rmse": None,
    }
    return jsonify({"status": "reset"})


@app.route("/api/prediction/new")
def get_new_prediction():
    if not workflow_status["completed"]:
        return jsonify({"error": "작업이 완료되지 않았습니다."}), 400

    stock = _get_stock_or_none()
    if not stock:
        return jsonify({"error": "선택된 종목이 없습니다."}), 404

    model, scaler, df = load_model_and_data(stock)
    if model is None:
        return jsonify({"error": "모델을 로드할 수 없습니다."}), 404

    prediction = predict_stock(model, scaler, df, stock)
    prediction["model_r2"] = workflow_status.get("model_r2") or 0
    prediction["model_rmse"] = workflow_status.get("model_rmse") or 0
    return jsonify(prediction)


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    logger.info("Flask 서버 시작 (http://0.0.0.0:%d, 로그: %s)", port, LOG_FILE)
    app.run(debug=True, host="0.0.0.0", port=port)
