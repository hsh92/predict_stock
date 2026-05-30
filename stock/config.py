"""종목별 경로 및 현재 선택 종목 관리."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

CURRENT_STOCK_FILE = Path("data/current_stock.json")
_lock = threading.Lock()


@dataclass
class StockContext:
    ticker: str
    name: str
    exchange: str = ""

    @property
    def slug(self) -> str:
        safe = re.sub(r"[^\w.-]", "_", self.ticker)
        return safe.replace(".", "_").replace("-", "_")

    @property
    def raw_file(self) -> Path:
        return Path(f"data/{self.slug}/5y.csv")

    @property
    def train_file(self) -> Path:
        return Path(f"data/processed/{self.slug}_train.csv")

    @property
    def test_file(self) -> Path:
        return Path(f"data/processed/{self.slug}_test.csv")

    @property
    def cleaned_file(self) -> Path:
        return Path(f"data/processed/{self.slug}_cleaned.csv")

    @property
    def model_dir(self) -> Path:
        return Path(f"models/{self.slug}")

    @property
    def model_file(self) -> Path:
        return self.model_dir / "linear_regression_model.pkl"

    @property
    def scaler_file(self) -> Path:
        return self.model_dir / "feature_scaler.pkl"

    @property
    def model_info_file(self) -> Path:
        return self.model_dir / "model_info.json"

    def ensure_dirs(self) -> None:
        self.raw_file.parent.mkdir(parents=True, exist_ok=True)
        self.train_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StockContext:
        return cls(
            ticker=data["ticker"],
            name=data.get("name", data["ticker"]),
            exchange=data.get("exchange", ""),
        )


def get_current_stock() -> StockContext | None:
    """사용자가 선택해 저장한 종목만 반환합니다 (레거시 자동 로드 없음)."""
    if not CURRENT_STOCK_FILE.exists():
        return None
    with _lock:
        data = json.loads(CURRENT_STOCK_FILE.read_text(encoding="utf-8"))
    return StockContext.from_dict(data)


def set_current_stock(stock: StockContext) -> None:
    CURRENT_STOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        CURRENT_STOCK_FILE.write_text(
            json.dumps(stock.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
