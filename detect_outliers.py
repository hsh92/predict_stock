#!/usr/bin/env python3
"""
삼성전자 주가 데이터 이상치 검사 및 정제 스크립트

- 무한대 값 검사
- NaN 값 검사
- IQR (Interquartile Range) 방법으로 이상치 탐지
- Z-score 방법으로 이상치 탐지
- 정제된 데이터 저장
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    sys.exit(1)


INPUT_FILE = Path("data/processed/samsung_005930_cleaned.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "samsung_005930_cleaned_outliers_removed.csv"


def load_data():
    """데이터를 로드합니다."""
    
    try:
        print(f"[*] 데이터 로드 중: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"[+] 로드 완료: {len(df)}행")
        return df
        
    except FileNotFoundError:
        print(f"[!] 파일을 찾을 수 없습니다: {INPUT_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def check_infinite_values(df):
    """무한대 값을 검사합니다."""
    
    print(f"\n[무한대 값 검사]")
    
    inf_found = False
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"  [주의] {col}: {inf_count}개 무한대 값 발견")
            inf_found = True
            # 무한대 값 제거
            df = df[~np.isinf(df[col])]
        else:
            print(f"  [OK] {col}: 무한대 값 없음")
    
    if not inf_found:
        print(f"  [OK] 모든 컬럼에서 무한대 값 없음")
    
    return df


def check_nan_values(df):
    """NaN 값을 검사합니다."""
    
    print(f"\n[NaN 값 검사]")
    
    nan_found = False
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"  [주의] {col}: {nan_count}개 NaN 값 발견")
            nan_found = True
        else:
            print(f"  [OK] {col}: NaN 값 없음")
    
    if not nan_found:
        print(f"  [OK] 모든 컬럼에서 NaN 값 없음")
    
    return df


def detect_outliers_iqr(df):
    """IQR (Interquartile Range) 방법으로 이상치를 탐지합니다."""
    
    print(f"\n[IQR 방법 이상치 탐지]")
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    outlier_rows = set()
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # IQR 범위를 벗어난 이상치 찾기
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = df[outlier_mask].index
        
        if len(outlier_indices) > 0:
            print(f"  - {col}:")
            print(f"    범위: [{lower_bound:,.2f}, {upper_bound:,.2f}]")
            print(f"    이상치 발견: {len(outlier_indices)}개")
            outlier_rows.update(outlier_indices)
        else:
            print(f"  [OK] {col}: 이상치 없음")
    
    return outlier_rows


def detect_outliers_zscore(df, threshold=3.0):
    """Z-score 방법으로 이상치를 탐지합니다.
    
    threshold: Z-score 임계값 (기본값 3.0 = 99.7%)
    """
    
    print(f"\n[Z-score 방법 이상치 탐지] (임계값: {threshold}σ)")
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    outlier_rows = set()
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        # Z-score 계산
        z_scores = np.abs(stats.zscore(df[col]))
        
        # 임계값을 초과하는 이상치 찾기
        outlier_mask = z_scores > threshold
        outlier_indices = df[outlier_mask].index
        
        if len(outlier_indices) > 0:
            print(f"  - {col}: {len(outlier_indices)}개 이상치 발견")
            print(f"    최대 Z-score: {z_scores.max():.2f}")
            outlier_rows.update(outlier_indices)
        else:
            print(f"  [OK] {col}: 이상치 없음")
    
    return outlier_rows


def detect_volume_anomalies(df):
    """거래량 이상 패턴을 검사합니다."""
    
    print(f"\n[거래량 이상 패턴 검사]")
    
    if 'Volume' not in df.columns:
        print(f"  Volume 컬럼 없음")
        return set()
    
    outlier_rows = set()
    
    # 거래량이 0인 경우
    zero_volume = df[df['Volume'] == 0].index
    if len(zero_volume) > 0:
        print(f"  [주의] 거래량 0: {len(zero_volume)}개")
        outlier_rows.update(zero_volume)
    else:
        print(f"  [OK] 거래량 0: 없음")
    
    # 거래량이 평균의 10배 이상인 경우 (극단적인 거래)
    mean_volume = df['Volume'].mean()
    extreme_volume = df[df['Volume'] > mean_volume * 10].index
    if len(extreme_volume) > 0:
        print(f"  [주의] 극단적 거래량 (평균의 10배 이상): {len(extreme_volume)}개")
        outlier_rows.update(extreme_volume)
    else:
        print(f"  [OK] 극단적 거래량: 없음")
    
    return outlier_rows


def detect_price_anomalies(df):
    """가격 이상 패턴을 검사합니다."""
    
    print(f"\n[가격 이상 패턴 검사]")
    
    outlier_rows = set()
    
    # High < Low 인 경우 (물리적으로 불가능)
    invalid_price = df[df['High'] < df['Low']].index
    if len(invalid_price) > 0:
        print(f"  [주의] 고가 < 저가 (논리 오류): {len(invalid_price)}개")
        outlier_rows.update(invalid_price)
    else:
        print(f"  [OK] 고가/저가 논리 검사: 통과")
    
    # Close가 High와 Low 사이에 없는 경우
    invalid_close = df[(df['Close'] > df['High']) | (df['Close'] < df['Low'])].index
    if len(invalid_close) > 0:
        print(f"  [주의] Close가 High/Low 범위 밖: {len(invalid_close)}개")
        outlier_rows.update(invalid_close)
    else:
        print(f"  [OK] 종가 범위 검사: 통과")
    
    # 하루 변동폭이 20% 이상인 경우
    daily_change_pct = abs((df['Close'] - df['Open']) / df['Open'] * 100)
    extreme_change = df[daily_change_pct > 20].index
    if len(extreme_change) > 0:
        print(f"  [주의] 극단적 일일 변동 (20% 이상): {len(extreme_change)}개")
        outlier_rows.update(extreme_change)
    else:
        print(f"  [OK] 극단적 변동: 없음")
    
    return outlier_rows


def print_outlier_details(df, outlier_rows):
    """이상치 상세 정보를 출력합니다."""
    
    if len(outlier_rows) == 0:
        print(f"\n[이상치 상세 정보]")
        print(f"  이상치가 없습니다.")
        return
    
    print(f"\n[이상치 상세 정보]")
    print(f"  총 {len(outlier_rows)}개 행에서 이상치 발견")
    
    outlier_df = df.loc[list(outlier_rows)].sort_index()
    
    print(f"\n  상위 10개:")
    for idx, (i, row) in enumerate(outlier_df.head(10).iterrows()):
        print(f"    {i+1}. {row['Date'].strftime('%Y-%m-%d')} - "
              f"O:{row['Open']:,.2f}, H:{row['High']:,.2f}, "
              f"L:{row['Low']:,.2f}, C:{row['Close']:,.2f}, "
              f"V:{row['Volume']:,.0f}")


def save_cleaned_data(df, original_count):
    """정제된 데이터를 저장합니다."""
    
    print(f"\n[정제된 데이터 저장]")
    
    removed_count = original_count - len(df)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print(f"  - 파일: {OUTPUT_FILE}")
    print(f"  - 원본 행 수: {original_count}")
    print(f"  - 제거된 행 수: {removed_count}")
    print(f"  - 정제 후 행 수: {len(df)}")
    print(f"  - 제거 비율: {removed_count/original_count*100:.2f}%")


def print_summary(df_original, df_cleaned):
    """정제 전후 비교 요약을 출력합니다."""
    
    print(f"\n{'='*70}")
    print(f"데이터 이상치 정제 완료")
    print(f"{'='*70}")
    
    print(f"\n[정제 결과]")
    print(f"  - 원본 행 수: {len(df_original)}")
    print(f"  - 정제 후 행 수: {len(df_cleaned)}")
    print(f"  - 제거된 행 수: {len(df_original) - len(df_cleaned)}")
    print(f"  - 유지율: {len(df_cleaned)/len(df_original)*100:.2f}%")
    
    print(f"\n[정제 전후 통계 비교]")
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df_original.columns:
            continue
        
        print(f"\n  {col}:")
        print(f"    정제 전:")
        print(f"      평균: {df_original[col].mean():,.2f}")
        print(f"      표준편차: {df_original[col].std():,.2f}")
        print(f"      범위: [{df_original[col].min():,.2f}, {df_original[col].max():,.2f}]")
        
        print(f"    정제 후:")
        print(f"      평균: {df_cleaned[col].mean():,.2f}")
        print(f"      표준편차: {df_cleaned[col].std():,.2f}")
        print(f"      범위: [{df_cleaned[col].min():,.2f}, {df_cleaned[col].max():,.2f}]")
    
    print(f"\n[사용 방법]")
    print(f"  정제된 데이터는 다음 경로에 저장됩니다:")
    print(f"  {OUTPUT_FILE}")
    print(f"\n  새로운 훈련/테스트 분할을 원하면:")
    print(f"  uv run python preprocess_samsung_stock.py")
    print(f"  (INPUT_FILE을 갱신하면 됩니다)")
    
    print(f"\n{'='*70}\n")


def main():
    """메인 함수: 이상치 검사 및 정제 파이프라인"""
    
    try:
        # 1. 데이터 로드
        df = load_data()
        original_count = len(df)
        
        # 2. 무한대 값 검사
        df = check_infinite_values(df)
        
        # 3. NaN 값 검사
        df = check_nan_values(df)
        
        # 4. IQR 방법 이상치 탐지
        outliers_iqr = detect_outliers_iqr(df)
        
        # 5. Z-score 방법 이상치 탐지
        outliers_zscore = detect_outliers_zscore(df, threshold=3.0)
        
        # 6. 거래량 이상 패턴 검사
        outliers_volume = detect_volume_anomalies(df)
        
        # 7. 가격 이상 패턴 검사
        outliers_price = detect_price_anomalies(df)
        
        # 모든 이상치 통합
        all_outliers = outliers_iqr | outliers_zscore | outliers_volume | outliers_price
        
        print(f"\n[통합 이상치 분석]")
        print(f"  - IQR 이상치: {len(outliers_iqr)}개")
        print(f"  - Z-score 이상치: {len(outliers_zscore)}개")
        print(f"  - 거래량 이상: {len(outliers_volume)}개")
        print(f"  - 가격 이상: {len(outliers_price)}개")
        print(f"  - 총 이상치 (중복 제거): {len(all_outliers)}개")
        
        # 이상치 상세 정보 출력
        print_outlier_details(df, all_outliers)
        
        # 8. 이상치 제거
        df_cleaned = df.drop(list(all_outliers)).reset_index(drop=True)
        
        # 9. 정제된 데이터 저장
        save_cleaned_data(df_cleaned, original_count)
        
        # 10. 요약 출력
        df_original = load_data()
        print_summary(df_original, df_cleaned)
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
