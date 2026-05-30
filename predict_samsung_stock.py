#!/usr/bin/env python3
"""
삼성전자 주가 예측 스크립트

학습된 선형 회귀 모델을 이용해 내일의 주가를 예측합니다.
"""

import sys
from pathlib import Path
import pickle

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    sys.exit(1)


MODEL_FILE = Path("models/linear_regression_model.pkl")
SCALER_FILE = Path("models/feature_scaler.pkl")
TEST_FILE = Path("data/processed/samsung_005930_test.csv")


def load_model_and_scaler():
    """저장된 모델과 스케일러를 로드합니다."""
    
    try:
        print(f"[*] 모델 로드 중...")
        
        if not MODEL_FILE.exists():
            print(f"[!] 모델 파일을 찾을 수 없습니다: {MODEL_FILE}")
            print("[!] train_linear_regression.py를 먼저 실행하세요.")
            sys.exit(1)
        
        # 모델 로드
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        # 스케일러 로드
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"[+] 모델 로드 완료")
        return model, scaler
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def predict_next_close(model, scaler, open_price, high_price, low_price, volume):
    """내일의 종가를 예측합니다.
    
    Args:
        model: 학습된 선형 회귀 모델
        scaler: 특징 스케일러
        open_price: 오늘의 시가
        high_price: 오늘의 고가
        low_price: 오늘의 저가
        volume: 오늘의 거래량
    
    Returns:
        예측된 내일의 종가
    """
    
    # 입력 데이터 생성
    input_features = np.array([[open_price, high_price, low_price, volume]])
    
    # 스케일링
    features_scaled = scaler.transform(input_features)
    
    # 예측
    predicted_close = model.predict(features_scaled)[0]
    
    return predicted_close


def predict_from_latest_data(model, scaler):
    """테스트 데이터의 최신 거래일 기반으로 내일 가격을 예측합니다."""
    
    print(f"\n[최신 거래일 기반 예측]")
    
    try:
        test_df = pd.read_csv(TEST_FILE)
        latest_row = test_df.iloc[-1]
        
        # 예측에 필요한 데이터
        open_price = latest_row['Open']
        high_price = latest_row['High']
        low_price = latest_row['Low']
        volume = latest_row['Volume']
        actual_close = latest_row['Close']
        
        # 예측
        predicted_close = predict_next_close(
            model, scaler,
            open_price, high_price, low_price, volume
        )
        
        print(f"  마지막 거래일: {latest_row['Date']}")
        print(f"  - 시가: {open_price:,.2f} 원")
        print(f"  - 고가: {high_price:,.2f} 원")
        print(f"  - 저가: {low_price:,.2f} 원")
        print(f"  - 거래량: {volume:,.0f}")
        print(f"  - 실제 종가: {actual_close:,.2f} 원")
        print(f"\n  [예측 결과]")
        print(f"  - 예측 종가: {predicted_close:,.2f} 원")
        print(f"  - 예상 변동: {predicted_close - actual_close:+,.2f} 원 ({(predicted_close - actual_close)/actual_close*100:+.2f}%)")
        
        return predicted_close
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        return None


def predict_from_user_input(model, scaler):
    """사용자가 입력한 데이터로부터 예측합니다."""
    
    print(f"\n[사용자 입력 기반 예측]")
    print(f"오늘의 주가 정보를 입력하세요:")
    
    try:
        open_price = float(input("  시가 (Open): "))
        high_price = float(input("  고가 (High): "))
        low_price = float(input("  저가 (Low): "))
        volume = float(input("  거래량 (Volume): "))
        
        # 예측
        predicted_close = predict_next_close(
            model, scaler,
            open_price, high_price, low_price, volume
        )
        
        print(f"\n  [입력 데이터]")
        print(f"  - 시가: {open_price:,.2f} 원")
        print(f"  - 고가: {high_price:,.2f} 원")
        print(f"  - 저가: {low_price:,.2f} 원")
        print(f"  - 거래량: {volume:,.0f}")
        print(f"\n  [예측 결과]")
        print(f"  - 내일 예측 종가: {predicted_close:,.2f} 원")
        
        return predicted_close
        
    except ValueError as e:
        print(f"[!] 입력 오류: 숫자를 입력해주세요.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        return None


def batch_predict(model, scaler, test_df):
    """테스트 데이터 전체에 대해 예측을 수행하고 성능을 평가합니다."""
    
    print(f"\n[배치 예측 (테스트 데이터)]")
    
    predictions = []
    actuals = []
    errors = []
    
    for idx, row in test_df.iterrows():
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        volume = row['Volume']
        actual_close = row['Close']
        
        # 예측
        predicted_close = predict_next_close(
            model, scaler,
            open_price, high_price, low_price, volume
        )
        
        predictions.append(predicted_close)
        actuals.append(actual_close)
        errors.append(abs(predicted_close - actual_close))
    
    # 성능 지표 계산
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    # 정확도 계산 (±500원 이내)
    accurate_count = sum(1 for e in errors if e <= 500)
    accuracy_rate = accurate_count / len(errors) * 100
    
    print(f"  - 예측 샘플: {len(predictions)}개")
    print(f"  - 평균 오차: {mean_error:,.2f} 원")
    print(f"  - 표준편차: {std_error:,.2f} 원")
    print(f"  - 최대 오차: {max_error:,.2f} 원")
    print(f"  - 최소 오차: {min_error:,.2f} 원")
    print(f"  - 정확도 (±500원): {accuracy_rate:.1f}%")
    
    return predictions, actuals, errors


def print_usage_guide():
    """모델 사용 가이드를 출력합니다."""
    
    print(f"\n{'='*70}")
    print(f"선형 회귀 모델 사용 가이드")
    print(f"{'='*70}\n")
    
    print(f"[Python에서 직접 사용]")
    print(f"""
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# 모델과 스케일러 로드
with open('models/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 데이터 준비 (시가, 고가, 저가, 거래량)
today_features = np.array([[310000, 320000, 305000, 40000000]])

# 특징 스케일링
features_scaled = scaler.transform(today_features)

# 내일 종가 예측
predicted_close = model.predict(features_scaled)[0]

print(f"내일 예측 종가: {{predicted_close:,.2f}} 원")
    """)
    
    print(f"\n[모델 해석]")
    print(f"  - 시가(Open) 계수: -5232.75")
    print(f"    → 시가가 올라도 종가는 약간 내려가는 경향")
    print(f"  - 고가(High) 계수: +6404.62")
    print(f"    → 고가가 높을수록 종가도 높음")
    print(f"  - 저가(Low) 계수: +6798.20")
    print(f"    → 저가가 높을수록 종가도 높음")
    print(f"  - 거래량(Volume) 계수: +0.68")
    print(f"    → 거래량은 종가에 미미한 양의 영향")
    
    print(f"\n{'='*70}\n")


def main():
    """메인 함수: 예측 인터페이스"""
    
    try:
        # 모델과 스케일러 로드
        model, scaler = load_model_and_scaler()
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(TEST_FILE)
        
        print(f"\n[메뉴]")
        print(f"1. 최신 거래일 기반 예측 (자동)")
        print(f"2. 사용자 입력 기반 예측")
        print(f"3. 배치 예측 (테스트 데이터 전체)")
        print(f"4. 사용 가이드")
        print(f"5. 종료")
        
        choice = input(f"\n선택 (1-5): ").strip()
        
        if choice == '1':
            predict_from_latest_data(model, scaler)
        elif choice == '2':
            predict_from_user_input(model, scaler)
        elif choice == '3':
            batch_predict(model, scaler, test_df)
        elif choice == '4':
            print_usage_guide()
        elif choice == '5':
            print(f"[*] 프로그램 종료")
            return 0
        else:
            print(f"[!] 잘못된 선택입니다.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
