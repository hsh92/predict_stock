#!/usr/bin/env python3
"""
삼성전자 주가 모델 관리 및 학습 스크립트

기능:
1. 저장된 모델 로드 (있는 경우)
2. 새로운 모델 학습 (선택사항)
3. 모델 성능 비교 및 평가
4. 더 좋은 모델 자동 선택 및 저장
"""

import sys
from pathlib import Path
import pickle
import json
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    sys.exit(1)


TRAIN_FILE = Path("data/processed/samsung_005930_train.csv")
TEST_FILE = Path("data/processed/samsung_005930_test.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "linear_regression_model.pkl"
SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json"


def load_existing_model():
    """저장된 모델을 로드합니다."""
    
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        return None, None, None
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        
        model_info = None
        if MODEL_INFO_FILE.exists():
            with open(MODEL_INFO_FILE, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
        
        return model, scaler, model_info
        
    except Exception as e:
        print(f"[경고] 기존 모델 로드 실패: {e}")
        return None, None, None


def load_data():
    """훈련 및 테스트 데이터를 로드합니다."""
    
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        return train_df, test_df
        
    except FileNotFoundError as e:
        print(f"[!] 파일을 찾을 수 없습니다: {e}")
        print("[!] preprocess_samsung_stock.py를 먼저 실행하세요.")
        sys.exit(1)


def prepare_features(df):
    """데이터를 특징과 목표로 분리합니다."""
    
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features].values
    y = df['Close'].values
    
    return X, y


def train_new_model(X_train, y_train, X_test, y_test):
    """새로운 선형 회귀 모델을 학습합니다."""
    
    print(f"\n[*] 새로운 모델 학습 중...")
    
    # 특징 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 성능 평가
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    metrics = {
        'train_mse': mean_squared_error(y_train, train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'test_r2': r2_score(y_test, test_pred),
    }
    
    print(f"[+] 모델 학습 완료")
    print(f"\n  훈련 성능:")
    print(f"    - RMSE: {metrics['train_rmse']:,.2f} 원")
    print(f"    - R²: {metrics['train_r2']:.6f}")
    print(f"  테스트 성능:")
    print(f"    - RMSE: {metrics['test_rmse']:,.2f} 원")
    print(f"    - R²: {metrics['test_r2']:.6f}")
    
    return model, scaler, metrics


def evaluate_model(model, scaler, X_test, y_test):
    """모델의 성능을 평가합니다."""
    
    X_test_scaled = scaler.transform(X_test)
    test_pred = model.predict(X_test_scaled)
    
    metrics = {
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'test_r2': r2_score(y_test, test_pred),
    }
    
    return metrics


def compare_models(existing_metrics, new_metrics):
    """기존 모델과 새 모델을 비교합니다."""
    
    print(f"\n{'='*70}")
    print(f"모델 성능 비교")
    print(f"{'='*70}")
    
    print(f"\n[테스트 R² 점수 (높을수록 좋음)]")
    print(f"  기존 모델: {existing_metrics['test_r2']:.6f}")
    print(f"  새로운 모델: {new_metrics['test_r2']:.6f}")
    
    improvement = new_metrics['test_r2'] - existing_metrics['test_r2']
    if improvement > 0:
        print(f"  개선도: +{improvement:.6f} (더 좋음) ✓")
        better_model = "new"
    elif improvement < 0:
        print(f"  개선도: {improvement:.6f} (더 나쁨)")
        better_model = "existing"
    else:
        print(f"  개선도: 동일")
        better_model = "existing"
    
    print(f"\n[테스트 RMSE (낮을수록 좋음)]")
    print(f"  기존 모델: {existing_metrics['test_rmse']:,.2f} 원")
    print(f"  새로운 모델: {new_metrics['test_rmse']:,.2f} 원")
    
    improvement_rmse = existing_metrics['test_rmse'] - new_metrics['test_rmse']
    if improvement_rmse > 0:
        print(f"  개선도: -{improvement_rmse:,.2f} 원 (더 좋음) ✓")
    elif improvement_rmse < 0:
        print(f"  개선도: +{-improvement_rmse:,.2f} 원 (더 나쁨)")
    else:
        print(f"  개선도: 동일")
    
    print(f"\n{'='*70}")
    
    return better_model


def save_model(model, scaler, metrics):
    """모델과 스케일러를 저장합니다."""
    
    print(f"\n[*] 모델 저장 중...")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    # 스케일러 저장
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 모델 정보 저장
    model_info = {
        'saved_time': datetime.now().isoformat(),
        'test_r2': float(metrics.get('test_r2', 0)),
        'test_rmse': float(metrics.get('test_rmse', 0)),
        'test_mae': float(metrics.get('test_mae', 0)),
        'train_r2': float(metrics.get('train_r2', 0)),
        'train_rmse': float(metrics.get('train_rmse', 0)),
        'train_mae': float(metrics.get('train_mae', 0)),
    }
    
    with open(MODEL_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"[+] 모델 저장 완료")
    print(f"  - 모델 파일: {MODEL_FILE}")
    print(f"  - 스케일러 파일: {SCALER_FILE}")
    print(f"  - 정보 파일: {MODEL_INFO_FILE}")


def print_model_info(model_info):
    """저장된 모델 정보를 출력합니다."""
    
    if not model_info:
        return
    
    print(f"\n[저장된 모델 정보]")
    print(f"  저장 시간: {model_info.get('saved_time', 'N/A')}")
    print(f"  테스트 R²: {model_info.get('test_r2', 'N/A'):.6f}")
    print(f"  테스트 RMSE: {model_info.get('test_rmse', 'N/A'):,.2f} 원")


def main():
    """메인 함수: 모델 관리 파이프라인"""
    
    try:
        print(f"\n{'='*70}")
        print(f"삼성전자 주가 예측 모델 관리 시스템")
        print(f"{'='*70}")
        
        # 1. 기존 모델 확인
        print(f"\n[1단계] 기존 모델 확인 중...")
        existing_model, existing_scaler, existing_model_info = load_existing_model()
        
        if existing_model is not None:
            print(f"[+] 저장된 모델 발견!")
            print_model_info(existing_model_info)
        else:
            print(f"[!] 저장된 모델을 찾을 수 없습니다.")
        
        # 2. 사용자 입력 받기
        print(f"\n[2단계] 사용자 입력 받기")
        retrain = input(f"\n새로운 모델을 학습하시겠습니까? (y/n, 기본값: n): ").strip().lower()
        
        if retrain != 'y':
            print(f"\n[*] 기존 모델 사용")
            if existing_model is None:
                print(f"[!] 저장된 모델이 없습니다. 모델을 먼저 학습해주세요.")
                print(f"[!] 다음 명령을 실행하세요: uv run python train_linear_regression.py")
                return 1
            
            print(f"\n[완료] 기존 모델 재사용 준비 완료")
            return 0
        
        # 3. 데이터 로드
        print(f"\n[3단계] 데이터 로드 중...")
        train_df, test_df = load_data()
        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)
        print(f"[+] 데이터 로드 완료 (훈련: {len(X_train)}행, 테스트: {len(X_test)}행)")
        
        # 4. 새로운 모델 학습
        print(f"\n[4단계] 새로운 모델 학습")
        new_model, new_scaler, new_metrics = train_new_model(X_train, y_train, X_test, y_test)
        
        # 5. 모델 비교
        if existing_model is not None:
            print(f"\n[5단계] 모델 성능 비교")
            
            # 기존 모델 성능 평가
            existing_metrics = evaluate_model(existing_model, existing_scaler, X_test, y_test)
            
            # 모델 비교
            better_model = compare_models(existing_metrics, new_metrics)
            
            if better_model == "new":
                print(f"\n[결정] 새로운 모델이 더 좋습니다!")
                print(f"새로운 모델을 저장하겠습니다.")
                save_model(new_model, new_scaler, new_metrics)
            else:
                print(f"\n[결정] 기존 모델이 더 좋습니다.")
                print(f"기존 모델을 계속 사용합니다.")
        else:
            print(f"\n[5단계] 새 모델 저장")
            print(f"기존 모델이 없으므로 새 모델을 저장합니다.")
            save_model(new_model, new_scaler, new_metrics)
        
        print(f"\n{'='*70}")
        print(f"모델 관리 완료")
        print(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
