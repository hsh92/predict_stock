#!/usr/bin/env python3
"""
삼성전자 주가 내일 예측 스크립트

학습된 선형 회귀 모델을 사용하여 내일의 주가를 예측합니다.
최신 거래일 데이터를 자동으로 로드하고 예측을 수행합니다.
"""

import sys
from pathlib import Path
import pickle
from datetime import datetime, timedelta

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    sys.exit(1)


# 파일 경로
CLEANED_FILE = Path("data/processed/samsung_005930_cleaned.csv")
MODEL_FILE = Path("models/linear_regression_model.pkl")
SCALER_FILE = Path("models/feature_scaler.pkl")


def load_model_and_data():
    """모델과 데이터를 로드합니다."""
    
    try:
        # 모델 로드
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        # 스케일러 로드
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        
        # 데이터 로드
        df = pd.read_csv(CLEANED_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        return model, scaler, df
        
    except FileNotFoundError as e:
        print(f"[!] 파일을 찾을 수 없습니다: {e}")
        print("[!] 다음을 먼저 실행하세요:")
        print("    1. uv run python detect_outliers.py")
        print("    2. uv run python train_linear_regression.py")
        sys.exit(1)
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def predict_tomorrow_price(model, scaler, df):
    """내일의 주가를 예측합니다."""
    
    # 최신 거래일 데이터 추출
    latest_row = df.iloc[-1]
    
    # 예측에 필요한 특징 (Open, High, Low, Volume)
    features = np.array([[
        latest_row['Open'],
        latest_row['High'],
        latest_row['Low'],
        latest_row['Volume']
    ]])
    
    # 특징 스케일링
    features_scaled = scaler.transform(features)
    
    # 예측
    predicted_close = model.predict(features_scaled)[0]
    
    # 내일 날짜 계산
    today = latest_row['Date']
    tomorrow = today + timedelta(days=1)
    
    return {
        'today_date': today,
        'tomorrow_date': tomorrow,
        'today_open': latest_row['Open'],
        'today_high': latest_row['High'],
        'today_low': latest_row['Low'],
        'today_close': latest_row['Close'],
        'today_volume': latest_row['Volume'],
        'predicted_close': predicted_close,
        'expected_change': predicted_close - latest_row['Close'],
        'expected_change_pct': (predicted_close - latest_row['Close']) / latest_row['Close'] * 100
    }


def print_prediction_result(result):
    """예측 결과를 출력합니다."""
    
    print(f"\n{'='*70}")
    print(f"삼성전자 주가 예측 결과")
    print(f"{'='*70}\n")
    
    print(f"[오늘의 거래 정보]")
    print(f"  날짜: {result['today_date'].strftime('%Y-%m-%d (%A)')}")
    print(f"  시가: {result['today_open']:>12,.0f} 원")
    print(f"  고가: {result['today_high']:>12,.0f} 원")
    print(f"  저가: {result['today_low']:>12,.0f} 원")
    print(f"  종가: {result['today_close']:>12,.0f} 원")
    print(f"  거래량: {result['today_volume']:>12,.0f}")
    
    print(f"\n[내일의 주가 예측]")
    print(f"  예측 날짜: {result['tomorrow_date'].strftime('%Y-%m-%d (%A)')}")
    print(f"  예측 종가: {result['predicted_close']:>12,.2f} 원")
    
    # 변동 방향 표시
    change_symbol = "▲" if result['expected_change'] > 0 else "▼"
    if result['expected_change'] == 0:
        change_symbol = "▬"
    
    print(f"\n[변동 예상]")
    print(f"  예상 변동: {change_symbol} {result['expected_change']:+12,.2f} 원")
    print(f"  변동률: {change_symbol} {result['expected_change_pct']:+12.2f}%")
    
    # 예측 평가
    print(f"\n[평가]")
    if abs(result['expected_change_pct']) < 0.5:
        assessment = "거의 변동 없을 것으로 예상됩니다."
    elif result['expected_change_pct'] > 1.0:
        assessment = "상승세가 강할 것으로 예상됩니다. (강함)"
    elif result['expected_change_pct'] > 0.5:
        assessment = "약한 상승세가 예상됩니다."
    elif result['expected_change_pct'] < -1.0:
        assessment = "하락세가 강할 것으로 예상됩니다. (강함)"
    else:
        assessment = "약한 하락세가 예상됩니다."
    
    print(f"  {assessment}")
    
    # 주의사항
    print(f"\n[주의사항]")
    print(f"  - 이 예측은 선형 회귀 모델 기반입니다.")
    print(f"  - 실제 주가는 다양한 외부 요인에 영향을 받습니다.")
    print(f"  - 투자 결정은 신중하게 하시기 바랍니다.")
    print(f"  - 정확한 정보는 금융 전문가와 상담하세요.")
    
    print(f"\n{'='*70}\n")


def save_prediction_log(result):
    """예측 결과를 로그 파일에 저장합니다."""
    
    log_dir = Path("predictions")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일명: YYYY-MM-DD_predictions.txt
    log_file = log_dir / f"{result['today_date'].strftime('%Y-%m-%d')}_prediction.txt"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"삼성전자 주가 예측 결과\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"예측 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"[오늘의 거래 정보]\n")
        f.write(f"  날짜: {result['today_date'].strftime('%Y-%m-%d')}\n")
        f.write(f"  시가: {result['today_open']:,.0f} 원\n")
        f.write(f"  고가: {result['today_high']:,.0f} 원\n")
        f.write(f"  저가: {result['today_low']:,.0f} 원\n")
        f.write(f"  종가: {result['today_close']:,.0f} 원\n")
        f.write(f"  거래량: {result['today_volume']:,.0f}\n\n")
        
        f.write(f"[내일의 주가 예측]\n")
        f.write(f"  예측 날짜: {result['tomorrow_date'].strftime('%Y-%m-%d')}\n")
        f.write(f"  예측 종가: {result['predicted_close']:,.2f} 원\n")
        f.write(f"  예상 변동: {result['expected_change']:+,.2f} 원\n")
        f.write(f"  변동률: {result['expected_change_pct']:+.2f}%\n")
    
    return log_file


def main():
    """메인 함수"""
    
    try:
        print(f"\n[*] 모델과 데이터 로드 중...")
        model, scaler, df = load_model_and_data()
        print(f"[+] 로드 완료")
        
        print(f"\n[*] 내일 주가 예측 중...")
        result = predict_tomorrow_price(model, scaler, df)
        print(f"[+] 예측 완료")
        
        # 결과 출력
        print_prediction_result(result)
        
        # 로그 저장
        log_file = save_prediction_log(result)
        print(f"[+] 예측 로그 저장: {log_file}\n")
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
