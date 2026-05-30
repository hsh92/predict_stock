# predict_stock

Yahoo Finance에서 **종목을 검색·선택**하면 최근 5년 일봉 데이터를 자동 수집하고, 선형 회귀 모델로 내일 종가를 예측하며, Flask 웹 대시보드에서 5년 분석 차트와 밸류에이션 지표를 확인할 수 있는 Python 프로젝트입니다.

> **투자 참고용 도구**입니다. 예측 결과는 투자 권유가 아니며, 실제 투자 결정은 본인 책임입니다.

---

## 목차

1. [프로젝트 소개](#1-프로젝트-소개)
2. [주요 기능](#2-주요-기능)
3. [개발 이력](#3-개발-이력)
4. [사전 요구사항](#4-사전-요구사항)
5. [설치 방법](#5-설치-방법)
6. [사용 방법 (순서대로)](#6-사용-방법-순서대로)
7. [웹 대시보드 상세](#7-웹-대시보드-상세)
8. [CLI 스크립트 (레거시)](#8-cli-스크립트-레거시)
9. [API 엔드포인트](#9-api-엔드포인트)
10. [프로젝트 구조](#10-프로젝트-구조)
11. [로컬 생성 파일 (Git 미포함)](#11-로컬-생성-파일-git-미포함)
12. [주의사항](#12-주의사항)
13. [문제 해결](#13-문제-해결)
14. [참고 자료](#14-참고-자료)

---

## 1. 프로젝트 소개

| 항목 | 내용 |
|------|------|
| 프로젝트명 | **predict_stock** |
| 데이터 소스 | Yahoo Finance (비공식 API, `yfinance`) |
| 지원 종목 | Yahoo Finance 검색 가능한 전 세계 주식·ETF |
| 수집 기간 | 최근 5년 (약 1,200거래일) |
| 데이터 | OHLCV (Open, High, Low, Close, Volume) |
| 예측 모델 | 선형 회귀 (Linear Regression) |
| 인터페이스 | Flask 웹 대시보드 (권장) + CLI 스크립트 |

### 처리 흐름

```
종목 검색 → 종목 선택 → Yahoo Finance 5년 데이터 수집
    → 전처리(80/20 분할) → 모델 학습 → 성능 평가 → 저장
    → 내일 종가 예측 + 5년 분석 차트 표시
```

---

## 2. 주요 기능

### 웹 대시보드 (권장)

- **종목 검색**: 종목명, 티커, 6자리 종목코드 (예: `005930` → `005930.KS`)
- **원클릭 파이프라인**: 선택 시 데이터 수집 → 학습 → 예측 자동 실행
- **예측 결과**: 최근 거래 정보, 내일 예측 종가, 변동률, 최근 10일 테이블
- **5년 주가 분석 차트** (Chart.js):
  - 5년 종가 추세 (MA20 / MA60)
  - 거래량, 30일 롤링 변동성
  - 일간 수익률 분포, 월별 수익률
- **밸류에이션 지표**: PER, PBR, PSR, EV/EBITDA, 베타, 시가총액 등 (Yahoo Finance 제공 시)
- **리스크 지표**: 연환산 변동성, 최대 낙폭(MDD), Sharpe Ratio
- **지표 툴팁**: 지표명·차트에 마우스를 올리면 설명 표시
- **재학습**: 최신 5년 데이터로 모델 갱신

### CLI 스크립트 (레거시)

삼성전자 단일 종목용 초기 스크립트가 포함되어 있습니다. 범용 웹 UI 사용을 권장합니다.

---

## 3. 개발 이력

| 버전 | 내용 |
|------|------|
| **v1.0** | 범용 종목 검색·선택, 종목별 데이터/모델 경로, 5년 분석 차트, PER 등 밸류에이션, 지표 툴팁 |
| **v0.10** | 웹 기반 모델 재학습, 5단계 진행 모달, REST API, 백그라운드 스레드 처리 |
| **v0.9** | 가격 1,000단위 표시, 원 이하 절사 |
| **v0.8** | Flask 웹 대시보드, 반응형 UI, REST API |
| **v0.7** | `manage_model.py` — 모델 비교·재학습 선택 |
| **v0.6** | `predict_tomorrow.py` — 자동 내일 예측 |
| **v0.5** | `analyze_and_visualize.py` — 6종 분석 그래프 |
| **v0.4** | `detect_outliers.py` — IQR/Z-score 이상치 정제 |
| **v0.3** | 선형 회귀 모델 학습·예측 |
| **v0.2** | 데이터 전처리, 훈련/테스트 80/20 분할 |
| **v0.1** | Yahoo Finance 데이터 수집, CSV 저장 |

---

## 4. 사전 요구사항

| 항목 | 요구 사항 |
|------|-----------|
| Python | **3.11 이상** (권장 3.12, `.python-version` 참고) |
| UV | [설치 가이드](https://docs.astral.sh/uv/getting-started/installation/) (권장) |
| 네트워크 | Yahoo Finance 데이터 수집을 위한 인터넷 연결 |
| OS | Windows / macOS / Linux |

---

## 5. 설치 방법

### 5-1. 저장소 클론

```powershell
git clone https://github.com/hsh92/predict_stock.git
cd predict_stock
```

### 5-2. 가상환경 생성

```powershell
uv venv
```

### 5-3. 가상환경 활성화

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

> PowerShell 실행 정책 오류: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 5-4. 패키지 설치

저장소에 포함된 `requirements.txt`로 설치합니다.

```powershell
uv pip install -r requirements.txt
```

**설치 확인:**

```powershell
python -c "import flask, yfinance, sklearn, pandas; print('OK')"
```

### 5-5. `uv run` 사용 시 주의

`uv run python app.py`는 setuptools 패키지 탐색 오류가 날 수 있습니다. **가상환경 활성화 후 `python app.py`** 를 사용하세요.

---

## 6. 사용 방법 (순서대로)

### Step 1 — 서버 실행

```powershell
cd predict_stock
.venv\Scripts\Activate.ps1    # Windows
python app.py
```

정상 실행 시:

```
Flask 서버 시작 (http://0.0.0.0:5000, 로그: logs\app.log)
 * Running on http://127.0.0.1:5000
```

### Step 2 — 브라우저 접속

**http://localhost:5000**

처음 접속 시 제목은 **「주가 예측」** 만 표시됩니다 (종목 미선택 상태).

### Step 3 — 종목 검색

검색창에 아래 형식으로 입력 후 **검색** 클릭:

| 입력 예 | 설명 |
|---------|------|
| `Apple` / `AAPL` | 미국 주식 |
| `005930` | 한국 6자리 코드 → `005930.KS` 자동 변환 |
| `Samsung` | 삼성전자 관련 결과 |
| `Tesla` / `TSLA` | 테슬라 |

> 한글 종목명은 Yahoo Finance 검색 한계로 결과가 없을 수 있습니다. **6자리 코드** 또는 **영문명**을 사용하세요.

### Step 4 — 종목 선택

검색 결과에서 원하는 종목의 **선택** 버튼 클릭.

진행 모달에서 5단계가 자동 실행됩니다:

1. 데이터 수집 (Yahoo Finance 5년)
2. 데이터 전처리 (NaN 제거, 80/20 분할)
3. 모델 학습 (선형 회귀)
4. 모델 평가 (R², RMSE)
5. 모델 저장

### Step 5 — 예측 결과 확인

완료 후 **결과 보기** → 페이지 새로고침.

- 상단: **「{회사명} 주가 예측」** 으로 제목 변경
- 중단: 최근 거래 정보, 내일 예측 종가, 변동률
- 하단: **5년 주가 분석** 차트 및 PER·변동성 등 지표

### Step 6 — (선택) 재학습

**「최신 데이터로 재학습」** 버튼 → Yahoo Finance에서 최신 5년 데이터를 다시 받아 모델을 갱신합니다.

### Step 7 — 서버 종료

터미널에서 `Ctrl + C`

---

## 7. 웹 대시보드 상세

### 예측 섹션

| 항목 | 설명 |
|------|------|
| 최근 거래 정보 | 선택 종목의 최신 OHLCV |
| 내일 주가 예측 | 선형 회귀 기반 예측 종가·변동률 |
| 최근 10일 테이블 | 일별 시가/고가/저가/종가/거래량 |

### 5년 분석 섹션

| 차트 | 설명 |
|------|------|
| 5년 종가 추세 | 종가 + MA20 + MA60 |
| 거래량 | 일별 거래량 |
| 30일 롤링 변동성 | 연환산 변동성 추이 |
| 일간 수익률 분포 | 수익률 히스토그램 |
| 월별 수익률 | 월별 상승/하락 막대 |

**지표 카드:** 연환산 변동성, MDD, Sharpe Ratio, PER, PBR 등  
**툴팁:** 밑줄(점선) 지표·차트 제목·그래프 데이터에 마우스를 올리면 설명 표시

---

## 8. CLI 스크립트 (레거시)

삼성전자(`005930.KS`) 단일 종목용입니다. 웹 UI 사용을 권장합니다.

| 순서 | 스크립트 | 설명 |
|------|----------|------|
| 1 | `fetch_samsung_stock.py` | 5년 데이터 수집 |
| 2 | `preprocess_samsung_stock.py` | 전처리·분할 |
| 3 | `detect_outliers.py` | 이상치 정제 (선택) |
| 4 | `analyze_and_visualize.py` | 분석 그래프 (선택) |
| 5 | `manage_model.py` | 모델 학습·관리 |
| 6 | `predict_samsung_stock.py` | 대화형 예측 |
| 7 | `predict_tomorrow.py` | 자동 내일 예측 |

```powershell
python fetch_samsung_stock.py
python preprocess_samsung_stock.py
python manage_model.py
```

---

## 9. API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 메인 웹 페이지 |
| `GET` | `/api/stocks/search?q=` | 종목 검색 |
| `POST` | `/api/stocks/select` | 종목 선택 + 파이프라인 시작 |
| `GET` | `/api/stocks/current` | 현재 선택 종목 |
| `GET` | `/api/prediction` | 예측 JSON |
| `GET` | `/api/history?days=30` | 주가 이력 |
| `GET` | `/api/analysis` | 5년 분석·차트 JSON |
| `POST` | `/api/retrain/start` | 재학습 시작 |
| `GET` | `/api/workflow/status` | 작업 진행 상태 |

---

## 10. 프로젝트 구조

```
predict_stock/
├── stock/                    # 핵심 모듈
│   ├── config.py             # 종목별 경로, 현재 선택 종목
│   ├── search.py             # Yahoo Finance 종목 검색
│   ├── pipeline.py           # 수집·전처리·학습·예측
│   └── analytics.py          # 5년 분석·차트·밸류에이션
├── templates/
│   ├── index.html            # 웹 대시보드
│   └── error.html            # 오류 페이지
├── app.py                    # Flask 메인 (진입점)
├── fetch_samsung_stock.py    # [레거시] 데이터 수집
├── preprocess_samsung_stock.py
├── train_linear_regression.py
├── manage_model.py
├── predict_samsung_stock.py
├── predict_tomorrow.py
├── detect_outliers.py
├── analyze_and_visualize.py
├── pyproject.toml            # 프로젝트 메타데이터
├── .python-version
├── .gitignore
└── README.md
```

---

## 11. 로컬 생성 파일 (Git 미포함)

실행 시 아래 파일·폴더가 **로컬에만** 생성됩니다. GitHub에 올라가지 않습니다.

```
data/                         # 수집 CSV, current_stock.json
├── {티커_slug}/5y.csv
└── processed/{티커_slug}_*.csv

models/                       # 학습된 모델
└── {티커_slug}/
    ├── linear_regression_model.pkl
    ├── feature_scaler.pkl
    ├── model_info.json
    ├── analysis.json
    └── charts.json

logs/app.log                  # 실행 로그
analysis/                     # CLI 분석 PNG (레거시)
predictions/                  # 예측 로그 (레거시)
.venv/                        # 가상환경
requirements.txt              # Python 패키지 의존성 목록
```

**로그 확인:**

```powershell
Get-Content logs\app.log -Tail 50 -Wait
```

---

## 12. 주의사항

### 투자 및 법적 고지

- 본 프로젝트는 **교육·연구·참고용**입니다.
- 선형 회귀는 과거 추세만 반영하며, 뉴스·정책·실적 등 외부 요인을 고려하지 않습니다.
- **예측 결과는 투자 권유가 아닙니다.** 투자 손실에 대한 책임은 사용자에게 있습니다.
- 전문가 상담 없이 투자 결정을 내리지 마세요.

### Yahoo Finance API

- **비공식 API** (`yfinance`) — 공식 API가 아닙니다.
- 간헐적 수집 실패, Rate limiting, 데이터 지연·오류 가능
- PER·PBR 등 밸류에이션은 **모든 종목에 제공되지 않음**
- 티커 형식: 한국 KOSPI `005930.KS`, KOSDAQ `035720.KQ`, 미국 `AAPL`

### 모델 한계

- 당일 Open/High/Low/Volume → 당일 Close 예측 구조 (lag 미사용)
- R²가 높아도 미래 예측 정확도는 보장되지 않음
- 급등락·이벤트 구간에서 오차가 클 수 있음

### 보안

- Flask `debug=True`는 **개발용**입니다. 공인 인터넷에 그대로 노출하지 마세요.
- `.env` 등 API 키 파일은 Git에 커밋하지 마세요.

### Windows 환경

- 콘솔 cp949 인코딩 이슈 → `logs/app.log`(UTF-8)로 로그 확인 권장
- PowerShell에서 `&&` 대신 `;` 또는 줄바꿈 사용

---

## 13. 문제 해결

### `ModuleNotFoundError`

```powershell
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### `uv run python app.py` 빌드 오류

```
Multiple top-level packages discovered in a flat-layout
```

→ `python app.py` 로 직접 실행

### `Address already in use` (포트 5000)

```powershell
Get-Process python | Stop-Process -Force
python app.py
```

### 분석 차트가 보이지 않음

1. 종목 **선택** 후 파이프라인 완료 여부 확인
2. **최신 데이터로 재학습** 실행
3. `logs/app.log` 에서 오류 확인

### 종목 검색 결과 없음

- 6자리 코드 (`005930`) 또는 영문 티커 (`AAPL`) 사용
- 인터넷 연결 확인 후 재시도

### "데이터를 가져오지 못했습니다"

- Yahoo Finance 일시 장애 → 잠시 후 재시도
- 티커 형식 확인 (한국: `.KS` / `.KQ`)

---

## 14. 참고 자료

- [yfinance](https://github.com/ranaroussi/yfinance)
- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Chart.js](https://www.chartjs.org/)
- [UV](https://docs.astral.sh/uv/)
- [Yahoo Finance](https://finance.yahoo.com)

---

## 라이선스

MIT License

---

## 14. E2E 테스트 (Playwright)

### 설치

```powershell
uv pip install -r requirements-dev.txt
playwright install chromium
```

### 실행 (헤드리스)

```powershell
pytest tests/e2e/ -v
```

### 브라우저로 보며 실행

```powershell
pytest tests/e2e/ -v --headed
```

### 테스트 내용

- **LG CNS 검색**: 한글 종목명 `"LG CNS"` 검색 시 결과 정상 반환 확인
- **티커 검색**: 숫자 코드 `"064400"` 검색 시 결과 정상 반환 확인
- **빈 검색어**: 검색어 없이 검색 버튼 클릭 시 안내 메시지 표시
