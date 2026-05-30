#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA_RAW_FILE = Path("data/samsung_005930_5y.csv")

# 메타데이터 행 무시하고 로드
df = pd.read_csv(DATA_RAW_FILE, skiprows=[0, 1], na_values=[''])

print("로드된 컬럼:", df.columns.tolist())
print("데이터 형태:", df.shape)
print("\n첫 3행:")
print(df.head(3))

# 컬럼명 표준화
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={'price': 'date'})

print("\n표준화된 컬럼:", df.columns.tolist())

# Date 컬럼을 datetime으로 변환
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("Date 타입:", df['date'].dtype)

# 컬럼명 최종 표준화
df.columns = ['Date' if c == 'date' else c.title() for c in df.columns]

print("\n최종 컬럼:", df.columns.tolist())

# 필요한 컬럼만 선택
required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
missing = [col for col in required if col not in df.columns]
print(f"\n필수 컬럼: {required}")
print(f"누락된 컬럼: {missing}")

if not missing:
    print("\n✓ 모든 필수 컬럼이 있습니다!")
    print(f"데이터 준비 완료: {len(df)}행")
