#!/usr/bin/env python3
"""
삼성전자 주가 데이터 분석 및 시각화 스크립트

정제된 데이터의 통계, 트렌드, 분포를 분석하고 그래프로 시각화합니다.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"오류: 필수 라이브러리를 찾을 수 없습니다. {e}")
    sys.exit(1)


# 파일 경로 설정
ORIGINAL_FILE = Path("data/processed/samsung_005930_cleaned.csv")
CLEANED_FILE = Path("data/processed/samsung_005930_cleaned_outliers_removed.csv")
OUTPUT_DIR = Path("analysis")

# matplotlib 한글 폰트 설정
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_data():
    """원본과 정제된 데이터를 로드합니다."""
    
    try:
        print(f"[*] 데이터 로드 중...")
        
        df_original = pd.read_csv(ORIGINAL_FILE)
        df_cleaned = pd.read_csv(CLEANED_FILE)
        
        df_original['Date'] = pd.to_datetime(df_original['Date'])
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
        
        print(f"[+] 원본 데이터: {len(df_original)}행")
        print(f"[+] 정제된 데이터: {len(df_cleaned)}행")
        
        return df_original, df_cleaned
        
    except FileNotFoundError as e:
        print(f"[!] 파일을 찾을 수 없습니다: {e}")
        print("[!] detect_outliers.py를 먼저 실행하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def create_output_directory():
    """출력 디렉터리를 생성합니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] 출력 디렉터리: {OUTPUT_DIR}")


def plot_price_trend(df_original, df_cleaned):
    """가격 추이 그래프를 그립니다."""
    
    print(f"\n[*] 가격 추이 그래프 생성 중...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 원본 데이터
    axes[0].plot(df_original['Date'], df_original['Close'], 
                 linewidth=1.5, color='#FF6B6B', label='Close (원본)')
    axes[0].fill_between(df_original['Date'], df_original['Low'], df_original['High'], 
                         alpha=0.2, color='#FF6B6B')
    axes[0].set_title('원본 데이터 - 가격 추이 (High/Low 범위)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('가격 (원)', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 정제된 데이터
    axes[1].plot(df_cleaned['Date'], df_cleaned['Close'], 
                 linewidth=1.5, color='#4ECDC4', label='Close (정제됨)')
    axes[1].fill_between(df_cleaned['Date'], df_cleaned['Low'], df_cleaned['High'], 
                         alpha=0.2, color='#4ECDC4')
    axes[1].set_title('정제된 데이터 - 가격 추이 (High/Low 범위)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('날짜', fontsize=11)
    axes[1].set_ylabel('가격 (원)', fontsize=11)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "01_price_trend.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] 저장됨: {filepath}")
    plt.close()


def plot_price_distribution(df_original, df_cleaned):
    """가격 분포 히스토그램을 그립니다."""
    
    print(f"\n[*] 가격 분포 히스토그램 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 원본 Open
    axes[0, 0].hist(df_original['Open'], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('원본 - 시가 (Open) 분포', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('가격 (원)', fontsize=10)
    axes[0, 0].set_ylabel('빈도', fontsize=10)
    axes[0, 0].axvline(df_original['Open'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {df_original["Open"].mean():,.0f}')
    axes[0, 0].legend(fontsize=9)
    
    # 정제 Open
    axes[0, 1].hist(df_cleaned['Open'], bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('정제된 - 시가 (Open) 분포', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('가격 (원)', fontsize=10)
    axes[0, 1].set_ylabel('빈도', fontsize=10)
    axes[0, 1].axvline(df_cleaned['Open'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {df_cleaned["Open"].mean():,.0f}')
    axes[0, 1].legend(fontsize=9)
    
    # 원본 Close
    axes[1, 0].hist(df_original['Close'], bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('원본 - 종가 (Close) 분포', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('가격 (원)', fontsize=10)
    axes[1, 0].set_ylabel('빈도', fontsize=10)
    axes[1, 0].axvline(df_original['Close'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {df_original["Close"].mean():,.0f}')
    axes[1, 0].legend(fontsize=9)
    
    # 정제 Close
    axes[1, 1].hist(df_cleaned['Close'], bins=50, color='#F38181', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('정제된 - 종가 (Close) 분포', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('가격 (원)', fontsize=10)
    axes[1, 1].set_ylabel('빈도', fontsize=10)
    axes[1, 1].axvline(df_cleaned['Close'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {df_cleaned["Close"].mean():,.0f}')
    axes[1, 1].legend(fontsize=9)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "02_price_distribution.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] 저장됨: {filepath}")
    plt.close()


def plot_volume_comparison(df_original, df_cleaned):
    """거래량 비교 그래프를 그립니다."""
    
    print(f"\n[*] 거래량 비교 그래프 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 원본 거래량 시계열
    axes[0, 0].bar(df_original['Date'], df_original['Volume'], 
                   color='#FF6B6B', alpha=0.7, width=1)
    axes[0, 0].set_title('원본 - 거래량 시계열', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('거래량', fontsize=10)
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # 정제된 거래량 시계열
    axes[0, 1].bar(df_cleaned['Date'], df_cleaned['Volume'], 
                   color='#4ECDC4', alpha=0.7, width=1)
    axes[0, 1].set_title('정제된 - 거래량 시계열', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('거래량', fontsize=10)
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # 원본 거래량 분포
    axes[1, 0].hist(df_original['Volume'], bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('원본 - 거래량 분포', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('거래량', fontsize=10)
    axes[1, 0].set_ylabel('빈도', fontsize=10)
    axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # 정제된 거래량 분포
    axes[1, 1].hist(df_cleaned['Volume'], bins=50, color='#F38181', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('정제된 - 거래량 분포', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('거래량', fontsize=10)
    axes[1, 1].set_ylabel('빈도', fontsize=10)
    axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "03_volume_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] 저장됨: {filepath}")
    plt.close()


def plot_boxplot_comparison(df_original, df_cleaned):
    """박스플롯을 통한 이상치 비교를 그립니다."""
    
    print(f"\n[*] 박스플롯 비교 그래프 생성 중...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 원본 박스플롯
    df_original_box = df_original[['Open', 'High', 'Low', 'Close']].copy()
    df_original_box.boxplot(ax=axes[0])
    axes[0].set_title('원본 데이터 - 박스플롯 (이상치 표시)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('가격 (원)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 정제된 박스플롯
    df_cleaned_box = df_cleaned[['Open', 'High', 'Low', 'Close']].copy()
    df_cleaned_box.boxplot(ax=axes[1])
    axes[1].set_title('정제된 데이터 - 박스플롯 (이상치 제거)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('가격 (원)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "04_boxplot_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] 저장됨: {filepath}")
    plt.close()


def plot_statistics_comparison(df_original, df_cleaned):
    """통계 지표 비교 그래프를 그립니다."""
    
    print(f"\n[*] 통계 지표 비교 그래프 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 평균 비교
    stats_mean = pd.DataFrame({
        '원본': df_original[['Open', 'High', 'Low', 'Close']].mean(),
        '정제된': df_cleaned[['Open', 'High', 'Low', 'Close']].mean()
    })
    stats_mean.plot(kind='bar', ax=axes[0, 0], color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[0, 0].set_title('평균 가격 비교', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('가격 (원)', fontsize=10)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 표준편차 비교
    stats_std = pd.DataFrame({
        '원본': df_original[['Open', 'High', 'Low', 'Close']].std(),
        '정제된': df_cleaned[['Open', 'High', 'Low', 'Close']].std()
    })
    stats_std.plot(kind='bar', ax=axes[0, 1], color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[0, 1].set_title('표준편차 비교', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('표준편차 (원)', fontsize=10)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 최소/최대값 비교 (Close)
    close_stats = pd.DataFrame({
        '원본_최소': [df_original['Close'].min()],
        '원본_최대': [df_original['Close'].max()],
        '정제_최소': [df_cleaned['Close'].min()],
        '정제_최대': [df_cleaned['Close'].max()]
    }).T
    close_stats.plot(kind='barh', ax=axes[1, 0], legend=False, color='#95E1D3', alpha=0.8)
    axes[1, 0].set_title('Close 가격 범위 비교', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('가격 (원)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 데이터 행 수 비교
    row_counts = pd.Series({
        '원본': len(df_original),
        '정제된': len(df_cleaned)
    })
    colors = ['#FF6B6B', '#4ECDC4']
    axes[1, 1].bar(row_counts.index, row_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 1].set_title('데이터 행 수 비교', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('행 수', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 값 라벨 추가
    for i, (idx, val) in enumerate(row_counts.items()):
        axes[1, 1].text(i, val + 20, f'{int(val):,}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "05_statistics_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] 저장됨: {filepath}")
    plt.close()


def plot_daily_change(df_original, df_cleaned):
    """일일 변동률 분석 그래프를 그립니다."""
    
    print(f"\n[*] 일일 변동률 분석 그래프 생성 중...")
    
    # 일일 변동률 계산
    df_original_change = ((df_original['Close'] - df_original['Open']) / df_original['Open'] * 100)
    df_cleaned_change = ((df_cleaned['Close'] - df_cleaned['Open']) / df_cleaned['Open'] * 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 원본 일일 변동률 시계열
    axes[0, 0].plot(df_original['Date'], df_original_change, 
                    linewidth=1, color='#FF6B6B', alpha=0.7, marker='o', markersize=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].fill_between(df_original['Date'], df_original_change, 0, alpha=0.3, color='#FF6B6B')
    axes[0, 0].set_title('원본 - 일일 변동률 시계열', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('변동률 (%)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 정제된 일일 변동률 시계열
    axes[0, 1].plot(df_cleaned['Date'], df_cleaned_change, 
                    linewidth=1, color='#4ECDC4', alpha=0.7, marker='o', markersize=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 1].fill_between(df_cleaned['Date'], df_cleaned_change, 0, alpha=0.3, color='#4ECDC4')
    axes[0, 1].set_title('정제된 - 일일 변동률 시계열', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('변동률 (%)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 원본 일일 변동률 분포
    axes[1, 0].hist(df_original_change, bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('원본 - 일일 변동률 분포', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('변동률 (%)', fontsize=10)
    axes[1, 0].set_ylabel('빈도', fontsize=10)
    axes[1, 0].axvline(df_original_change.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {df_original_change.mean():.2f}%')
    axes[1, 0].legend(fontsize=9)
    
    # 정제된 일일 변동률 분포
    axes[1, 1].hist(df_cleaned_change, bins=50, color='#F38181', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('정제된 - 일일 변동률 분포', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('변동률 (%)', fontsize=10)
    axes[1, 1].set_ylabel('빈도', fontsize=10)
    axes[1, 1].axvline(df_cleaned_change.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {df_cleaned_change.mean():.2f}%')
    axes[1, 1].legend(fontsize=9)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / "06_daily_change.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] 저장됨: {filepath}")
    plt.close()


def print_analysis_summary(df_original, df_cleaned):
    """분석 요약을 출력합니다."""
    
    print(f"\n{'='*70}")
    print(f"데이터 분석 및 시각화 완료")
    print(f"{'='*70}")
    
    print(f"\n[분석 결과 요약]")
    
    print(f"\n1. 데이터 정제 효과")
    print(f"  - 원본 행 수: {len(df_original):,}")
    print(f"  - 정제 후 행 수: {len(df_cleaned):,}")
    print(f"  - 제거율: {(len(df_original)-len(df_cleaned))/len(df_original)*100:.2f}%")
    
    print(f"\n2. 가격 통계 (Close)")
    print(f"  원본:")
    print(f"    평균: {df_original['Close'].mean():,.2f} 원")
    print(f"    표준편차: {df_original['Close'].std():,.2f} 원")
    print(f"    최소/최대: {df_original['Close'].min():,.2f} / {df_original['Close'].max():,.2f}")
    print(f"  정제된:")
    print(f"    평균: {df_cleaned['Close'].mean():,.2f} 원")
    print(f"    표준편차: {df_cleaned['Close'].std():,.2f} 원")
    print(f"    최소/최대: {df_cleaned['Close'].min():,.2f} / {df_cleaned['Close'].max():,.2f}")
    
    print(f"\n3. 거래량 통계 (Volume)")
    print(f"  원본:")
    print(f"    평균: {df_original['Volume'].mean():,.0f}")
    print(f"    표준편차: {df_original['Volume'].std():,.0f}")
    print(f"  정제된:")
    print(f"    평균: {df_cleaned['Volume'].mean():,.0f}")
    print(f"    표준편차: {df_cleaned['Volume'].std():,.0f}")
    
    print(f"\n4. 일일 변동률 통계")
    change_original = ((df_original['Close'] - df_original['Open']) / df_original['Open'] * 100)
    change_cleaned = ((df_cleaned['Close'] - df_cleaned['Open']) / df_cleaned['Open'] * 100)
    print(f"  원본:")
    print(f"    평균: {change_original.mean():.4f}%")
    print(f"    표준편차: {change_original.std():.4f}%")
    print(f"  정제된:")
    print(f"    평균: {change_cleaned.mean():.4f}%")
    print(f"    표준편차: {change_cleaned.std():.4f}%")
    
    print(f"\n[생성된 그래프]")
    print(f"  01_price_trend.png - 가격 추이 비교")
    print(f"  02_price_distribution.png - 가격 분포 히스토그램")
    print(f"  03_volume_comparison.png - 거래량 비교")
    print(f"  04_boxplot_comparison.png - 박스플롯 이상치 비교")
    print(f"  05_statistics_comparison.png - 통계 지표 비교")
    print(f"  06_daily_change.png - 일일 변동률 분석")
    
    print(f"\n[그래프 저장 경로]")
    print(f"  {OUTPUT_DIR.resolve()}")
    
    print(f"\n{'='*70}\n")


def main():
    """메인 함수: 데이터 분석 및 시각화 파이프라인"""
    
    try:
        # 1. 데이터 로드
        df_original, df_cleaned = load_data()
        
        # 2. 출력 디렉터리 생성
        create_output_directory()
        
        # 3. 그래프 생성
        plot_price_trend(df_original, df_cleaned)
        plot_price_distribution(df_original, df_cleaned)
        plot_volume_comparison(df_original, df_cleaned)
        plot_boxplot_comparison(df_original, df_cleaned)
        plot_statistics_comparison(df_original, df_cleaned)
        plot_daily_change(df_original, df_cleaned)
        
        # 4. 분석 요약 출력
        print_analysis_summary(df_original, df_cleaned)
        
        return 0
        
    except Exception as e:
        print(f"[!] 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
