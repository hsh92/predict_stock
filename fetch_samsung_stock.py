#!/usr/bin/env python3
"""
삼성전자(005930.KS) Yahoo Finance 데이터 수집 스크립트

최근 5년치 일봉 데이터(OHLCV)를 가져와 CSV로 저장합니다.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    print("다음 명령어로 설치하세요: uv pip install -r requirements.txt")
    sys.exit(1)


TICKER = "005930.KS"
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "samsung_005930_5y.csv"


def fetch_and_save():
    """Yahoo Finance에서 삼성전자 5년치 데이터를 수집해 CSV로 저장합니다."""
    
    try:
        print(f"[*] {TICKER} 데이터 수집 중...")
        
        # 5년 전 날짜 계산
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5 * 365)
        
        print(f"    기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # Yahoo Finance에서 데이터 다운로드
        df = yf.download(
            TICKER,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # 데이터 유효성 검증
        if df.empty:
            raise RuntimeError(
                f"데이터를 가져오지 못했습니다. "
                f"티커 '{TICKER}'가 올바른지 확인하세요."
            )
        
        print(f"[+] 데이터 수집 완료: {len(df)}개 거래일")
        
        # 출력 디렉터리 생성
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # CSV로 저장 (UTF-8-SIG 인코딩, 인덱스 포함)
        df.to_csv(OUTPUT_FILE, encoding="utf-8-sig")
        
        print(f"[+] CSV 파일 저장 완료: {OUTPUT_FILE.resolve()}")
        print(f"\n{'='*60}")
        print(f"데이터 요약")
        print(f"{'='*60}")
        
        # 기본 정보
        print(f"\n[기본 정보]")
        print(f"  - 첫 거래일: {df.index[0].strftime('%Y-%m-%d')}")
        print(f"  - 마지막 거래일: {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  - 전체 거래일: {len(df)}일")
        
        # 컬럼 출력 (MultiIndex 대응)
        columns_str = ", ".join([str(col) for col in df.columns])
        print(f"  - 컬럼: {columns_str}")
        
        # Close 가격 기본 통계
        close_col = [col for col in df.columns if 'Close' in str(col)][0]
        close_prices = df[close_col]
        
        print(f"\n[종가(Close) 통계]")
        print(f"  - 평균: {close_prices.mean():,.2f} 원")
        print(f"  - 표준편차: {close_prices.std():,.2f} 원")
        print(f"  - 최솟값: {close_prices.min():,.2f} 원")
        print(f"  - 최댓값: {close_prices.max():,.2f} 원")
        print(f"  - 중앙값: {close_prices.median():,.2f} 원")
        
        # 최근 5일 데이터
        recent_5days = df.tail(5)
        print(f"\n[최근 5일 데이터]")
        print(f"{'날짜':<12} {'시가':>10} {'고가':>10} {'저가':>10} {'종가':>10} {'거래량':>12}")
        print(f"{'-'*60}")
        
        for date, row in recent_5days.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            # MultiIndex 컬럼에서 필요한 값 추출
            open_col = [col for col in df.columns if 'Open' in str(col)][0]
            high_col = [col for col in df.columns if 'High' in str(col)][0]
            low_col = [col for col in df.columns if 'Low' in str(col)][0]
            vol_col = [col for col in df.columns if 'Volume' in str(col)][0]
            
            print(f"{date_str:<12} {row[open_col]:>10,.0f} {row[high_col]:>10,.0f} "
                  f"{row[low_col]:>10,.0f} {row[close_col]:>10,.0f} {row[vol_col]:>12,.0f}")
        
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(fetch_and_save())
