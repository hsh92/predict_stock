#!/usr/bin/env python3
"""
삼성전자 주가 선형 회귀 예측 모델

정제된 훈련 데이터를 이용해 내일의 주가를 예상하는 선형 회귀 모델을 학습합니다.
- 모델: 선형 회귀 (Linear Regression)
- 특징: Open, High, Low, Volume (이전 거래일의 가격과 거래량)
- 목표: Close (내일의 종가 예상)
"""

import sys
from pathlib import Path
import pickle
from datetime import datetime, timedelta

try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    print("다음 명령어로 설치하세요: uv pip install -r requirements.txt")
    sys.exit(1)


TRAIN_FILE = Path("data/processed/samsung_005930_train.csv")
TEST_FILE = Path("data/processed/samsung_005930_test.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "linear_regression_model.pkl"
SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"


def load_data():
    """훈련 데이터와 테스트 데이터를 로드합니다."""
    
    try:
        print(f"[*] 데이터 로드 중...")
        
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        print(f"[+] 훈련 데이터 로드: {len(train_df)}행")
        print(f"[+] 테스트 데이터 로드: {len(test_df)}행")
        
        return train_df, test_df
        
    except FileNotFoundError as e:
        print(f"[!] 파일을 찾을 수 없습니다: {e}")
        print("[!] preprocess_samsung_stock.py를 먼저 실행하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def prepare_features(df):
    """데이터를 특징과 목표로 분리합니다.
    
    특징(X): 시가, 고가, 저가, 거래량 (이전 거래일의 정보)
    목표(y): 종가 (다음 거래일의 종가, 즉 현재 행의 종가)
    """
    
    print(f"\n[데이터 준비]")
    
    # 특징 선택
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features].values
    
    # 목표 선택 (종가)
    y = df['Close'].values
    
    print(f"  - 특징 (X): {features}")
    print(f"  - 목표 (y): Close (종가)")
    print(f"  - 샘플 수: {len(X)}개")
    
    return X, y, features


def train_model(X_train, y_train, X_test, y_test):
    """선형 회귀 모델을 학습합니다."""
    
    print(f"\n[모델 학습]")
    
    # 특징 스케일링 (정규화)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  - 특징 스케일링 완료")
    
    # 선형 회귀 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print(f"  - 모델 학습 완료")
    
    # 모델 성능 평가
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)
    
    # 훈련 데이터 성능
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    
    # 테스트 데이터 성능
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"\n[모델 성능 평가]")
    print(f"  훈련 데이터:")
    print(f"    - RMSE: {train_rmse:,.2f} 원")
    print(f"    - MAE: {train_mae:,.2f} 원")
    print(f"    - R² 점수: {train_r2:.4f}")
    
    print(f"  테스트 데이터:")
    print(f"    - RMSE: {test_rmse:,.2f} 원")
    print(f"    - MAE: {test_mae:,.2f} 원")
    print(f"    - R² 점수: {test_r2:.4f}")
    
    # 모델 계수 출력
    print(f"\n[모델 계수]")
    feature_names = ['Open', 'High', 'Low', 'Volume']
    for name, coef in zip(feature_names, model.coef_):
        print(f"  - {name}: {coef:.6f}")
    print(f"  - 절편(Intercept): {model.intercept_:,.2f} 원")
    
    return model, scaler, {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }


def save_model(model, scaler):
    """학습된 모델과 스케일러를 저장합니다."""
    
    print(f"\n[모델 저장]")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"  - 모델: {MODEL_FILE}")
    
    # 스케일러 저장
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  - 스케일러: {SCALER_FILE}")


def predict_next_close(model, scaler, latest_data):
    """최신 거래일 데이터를 이용해 내일의 종가를 예측합니다.
    
    Args:
        model: 학습된 선형 회귀 모델
        scaler: 특징 스케일러
        latest_data: 최신 거래일의 [Open, High, Low, Volume]
    
    Returns:
        예측된 내일의 종가
    """
    
    # 데이터 스케일링
    features_scaled = scaler.transform([latest_data])
    
    # 예측
    predicted_close = model.predict(features_scaled)[0]
    
    return predicted_close


def print_summary(model, scaler, test_df, metrics):
    """모델 학습 결과를 요약 출력합니다."""
    
    print(f"\n{'='*70}")
    print(f"선형 회귀 모델 학습 완료")
    print(f"{'='*70}")
    
    print(f"\n[모델 정보]")
    print(f"  - 알고리즘: Linear Regression (선형 회귀)")
    print(f"  - 특징 개수: 4개 (Open, High, Low, Volume)")
    print(f"  - 훈련 샘플: {len(test_df) // (1-0.8):.0f}개")
    
    print(f"\n[성능 지표]")
    print(f"  훈련 데이터:")
    print(f"    - RMSE (평균 제곱근 오차): {metrics['train_rmse']:,.2f} 원")
    print(f"    - MAE (평균 절대 오차): {metrics['train_mae']:,.2f} 원")
    print(f"    - R² (결정 계수): {metrics['train_r2']:.4f}")
    
    print(f"  테스트 데이터:")
    print(f"    - RMSE: {metrics['test_rmse']:,.2f} 원")
    print(f"    - MAE: {metrics['test_mae']:,.2f} 원")
    print(f"    - R² (결정 계수): {metrics['test_r2']:.4f}")
    
    # 테스트 데이터의 마지막 거래일로 내일 가격 예측
    print(f"\n[내일 주가 예측 예시]")
    latest_row = test_df.iloc[-1]
    latest_features = [latest_row['Open'], latest_row['High'], latest_row['Low'], latest_row['Volume']]
    
    predicted_close = predict_next_close(model, scaler, latest_features)
    actual_close = latest_row['Close']
    
    print(f"  마지막 거래일: {latest_row['Date']}")
    print(f"  - 시가: {latest_row['Open']:,.2f} 원")
    print(f"  - 고가: {latest_row['High']:,.2f} 원")
    print(f"  - 저가: {latest_row['Low']:,.2f} 원")
    print(f"  - 거래량: {latest_row['Volume']:,.0f}")
    print(f"  - 실제 종가: {actual_close:,.2f} 원")
    print(f"  - 예측 종가: {predicted_close:,.2f} 원")
    print(f"  - 오차: {abs(actual_close - predicted_close):,.2f} 원 ({abs(actual_close - predicted_close)/actual_close*100:.2f}%)")
    
    print(f"\n[모델 사용 방법]")
    print(f"""
import pickle
from sklearn.preprocessing import StandardScaler

# 모델과 스케일러 로드
with open('models/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 최신 거래일 데이터 (Open, High, Low, Volume)
latest_data = [최신_시가, 최신_고가, 최신_저가, 최신_거래량]

# 내일 종가 예측
features_scaled = scaler.transform([latest_data])
predicted_close = model.predict(features_scaled)[0]

print(f"예측 종가: {{predicted_close:,.2f}} 원")
    """)
    
    print(f"{'='*70}\n")


def main():
    """메인 함수: 모델 학습 파이프라인 실행"""
    
    try:
        # 1. 데이터 로드
        train_df, test_df = load_data()
        
        # 2. 특징 준비
        X_train, y_train, features = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)
        
        # 3. 모델 학습
        model, scaler, metrics = train_model(X_train, y_train, X_test, y_test)
        
        # 4. 모델 저장
        save_model(model, scaler)
        
        # 5. 요약 출력
        print_summary(model, scaler, test_df, metrics)
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
