#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA_RAW_FILE = Path("data/samsung_005930_5y.csv")

# 첫 2행을 스킵하고 로드
df = pd.read_csv(DATA_RAW_FILE, skiprows=2)

print("로드된 컬럼:", df.columns.tolist())
print("데이터 형태:", df.shape)
print("\n첫 3행:")
print(df.head(3))

# Price를 Date로 이름 변경
if 'Price' in df.columns:
    df = df.rename(columns={'Price': 'Date'})
    print("\n[✓] Price -> Date로 이름 변경")

print("변경된 컬럼:", df.columns.tolist())

# Date를 datetime으로 변환
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print("\n[✓] Date를 datetime으로 변환")

# 필수 컬럼
required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
available = [c for c in required if c in df.columns]
missing = [c for c in required if c not in df.columns]

print(f"\n필수 컬럼: {required}")
print(f"사용 가능: {available}")
print(f"누락: {missing}")

if not missing:
    print("\n✓✓✓ 모든 컬럼이 준비되었습니다!")
else:
    print("\n✗ 컬럼 누락!")
