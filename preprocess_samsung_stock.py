#!/usr/bin/env python3
"""
삼성전자 주가 데이터 전처리 스크립트

- 누락된 데이터(빈 칸) 제거
- 훈련용(train)과 테스트용(test) 데이터로 분할
- 분할된 데이터를 CSV 파일로 저장
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    print("다음 명령어로 설치하세요: uv pip install -r requirements.txt")
    sys.exit(1)


INPUT_FILE = Path("data/samsung_005930_5y.csv")
OUTPUT_DIR = Path("data/processed")

# 훈련/테스트 분할 비율 (80% 훈련, 20% 테스트)
TRAIN_RATIO = 0.8


def load_and_inspect_data():
    """CSV 파일을 로드하고 데이터 상태를 검사합니다."""
    
    try:
        print(f"[*] 데이터 파일 로드 중: {INPUT_FILE}")
        
        # yfinance로 저장된 CSV 파일 로드
        # 첫 2행(Price, Ticker와 Date, NaN)을 제거하고 실제 데이터만 로드
        df = pd.read_csv(INPUT_FILE, skiprows=[0, 1])
        
        # 컬럼명 지정: Date, Close, High, Low, Open, Volume
        # (yfinance의 출력 순서: Close, High, Low, Open, Volume)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Date를 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 컬럼 순서 재정렬 (Date, Open, High, Low, Close, Volume)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"[+] 로드 완료: {len(df)}행")
        
        # 데이터 상태 검사
        print(f"\n[데이터 상태 검사]")
        print(f"  - 전체 행: {len(df)}행")
        print(f"  - 컬럼: {list(df.columns)}")
        print(f"  - 데이터 타입:")
        for col in df.columns:
            print(f"    - {col}: {df[col].dtype}")
        
        # 누락된 값 확인
        missing_counts = df.isnull().sum()
        print(f"\n[누락된 데이터]")
        total_missing = missing_counts.sum()
        if total_missing == 0:
            print(f"  - 누락된 데이터 없음")
        else:
            for col in df.columns:
                if missing_counts[col] > 0:
                    print(f"  - {col}: {missing_counts[col]}행 ({missing_counts[col]/len(df)*100:.2f}%)")
            print(f"  - 총 누락: {total_missing}행")
        
        return df
        
    except FileNotFoundError:
        print(f"[!] 파일을 찾을 수 없습니다: {INPUT_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def clean_data(df):
    """누락된 데이터를 제거합니다."""
    
    print(f"\n[데이터 정제]")
    print(f"  - 정제 전: {len(df)}행")
    
    # 누락된 값 제거
    df_clean = df.dropna()
    
    rows_removed = len(df) - len(df_clean)
    print(f"  - 제거된 행: {rows_removed}행")
    print(f"  - 정제 후: {len(df_clean)}행")
    
    # 인덱스 초기화
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def split_train_test(df, train_ratio=0.8):
    """데이터를 훈련용과 테스트용으로 분할합니다."""
    
    print(f"\n[훈련/테스트 분할]")
    print(f"  - 분할 비율: {train_ratio*100:.0f}% (훈련) / {(1-train_ratio)*100:.0f}% (테스트)")
    
    # 시계열 데이터이므로 시간순으로 분할 (무작위 분할 X)
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"  - 훈련 데이터: {len(train_df)}행 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - 테스트 데이터: {len(test_df)}행 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 기간 출력
    print(f"\n[데이터 기간]")
    print(f"  - 훈련: {train_df['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {train_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  - 테스트: {test_df['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {test_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    return train_df, test_df


def save_processed_data(train_df, test_df):
    """전처리된 데이터를 저장합니다."""
    
    print(f"\n[데이터 저장]")
    
    # 출력 디렉터리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 훈련 데이터 저장
    train_file = OUTPUT_DIR / "samsung_005930_train.csv"
    train_df.to_csv(train_file, index=False, encoding="utf-8-sig")
    print(f"  - 훈련 데이터: {train_file}")
    
    # 테스트 데이터 저장
    test_file = OUTPUT_DIR / "samsung_005930_test.csv"
    test_df.to_csv(test_file, index=False, encoding="utf-8-sig")
    print(f"  - 테스트 데이터: {test_file}")
    
    # 전체 정제된 데이터 저장 (참고용)
    full_clean = pd.concat([train_df, test_df], ignore_index=True)
    full_file = OUTPUT_DIR / "samsung_005930_cleaned.csv"
    full_clean.to_csv(full_file, index=False, encoding="utf-8-sig")
    print(f"  - 정제된 전체 데이터: {full_file}")
    
    return train_file, test_file, full_file


def print_summary(train_df, test_df, full_clean_df):
    """전처리 결과를 요약 출력합니다."""
    
    print(f"\n{'='*70}")
    print(f"전처리 완료 요약")
    print(f"{'='*70}")
    
    # 파일 정보
    print(f"\n[생성된 파일]")
    print(f"  - 훈련 데이터 파일: data/processed/samsung_005930_train.csv")
    print(f"  - 테스트 데이터 파일: data/processed/samsung_005930_test.csv")
    print(f"  - 정제된 전체 데이터: data/processed/samsung_005930_cleaned.csv")
    
    # 데이터 통계
    print(f"\n[데이터 통계]")
    print(f"  - 훈련 데이터: {len(train_df):,}행")
    print(f"  - 테스트 데이터: {len(test_df):,}행")
    print(f"  - 전체: {len(full_clean_df):,}행")
    
    # 컬럼 정보
    print(f"\n[컬럼 정보]")
    for col in train_df.columns:
        print(f"  - {col}")
    
    # 통계 정보
    print(f"\n[종가(Close) 통계]")
    
    print(f"  훈련 데이터:")
    print(f"    - 평균: {train_df['Close'].mean():,.2f} 원")
    print(f"    - 최소: {train_df['Close'].min():,.2f} 원")
    print(f"    - 최대: {train_df['Close'].max():,.2f} 원")
    print(f"    - 표준편차: {train_df['Close'].std():,.2f} 원")
    
    print(f"  테스트 데이터:")
    print(f"    - 평균: {test_df['Close'].mean():,.2f} 원")
    print(f"    - 최소: {test_df['Close'].min():,.2f} 원")
    print(f"    - 최대: {test_df['Close'].max():,.2f} 원")
    print(f"    - 표준편차: {test_df['Close'].std():,.2f} 원")
    
    print(f"\n[훈련/테스트 분할 확인]")
    print(f"  - 훈련 기간: {train_df['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {train_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  - 테스트 기간: {test_df['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {test_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  - 훈련 비율: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}%")
    print(f"  - 테스트 비율: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")
    
    print(f"\n[사용 예시 (Python)]")
    print(f"  import pandas as pd")
    print(f"  train = pd.read_csv('data/processed/samsung_005930_train.csv')")
    print(f"  test = pd.read_csv('data/processed/samsung_005930_test.csv')")
    print(f"\n{'='*70}\n")


def main():
    """메인 함수: 전처리 파이프라인 실행"""
    
    try:
        # 1. 데이터 로드 및 검사
        df = load_and_inspect_data()
        
        # 2. 데이터 정제
        df_clean = clean_data(df)
        
        # 3. 훈련/테스트 분할
        train_df, test_df = split_train_test(df_clean, TRAIN_RATIO)
        
        # 4. 데이터 저장
        train_file, test_file, full_file = save_processed_data(train_df, test_df)
        
        # 5. 요약 출력
        full_clean = pd.concat([train_df, test_df], ignore_index=True)
        print_summary(train_df, test_df, full_clean)
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
