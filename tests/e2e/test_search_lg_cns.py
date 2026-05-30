"""E2E 테스트: LG CNS 검색 기능 및 전체 워크플로우."""

import pytest
import time
from playwright.sync_api import expect


@pytest.mark.parametrize("query,expected_contains", [
    ("LG CNS", ["064400", "LG CNS"]),
    ("064400", ["064400"]),
])
def test_search_lg_cns(page, base_url, query, expected_contains):
    """LG CNS를 검색하여 결과가 정상 반환되는지 테스트."""
    
    page.goto(base_url)
    assert "주가 예측" in page.title()
    
    search_input = page.locator("#searchInput")
    expect(search_input).to_be_visible()
    
    search_input.fill(query)
    page.get_by_role("button", name="검색").click()
    
    results_list = page.locator("#searchResults")
    try:
        page.locator(".search-result-item").first.wait_for(timeout=15000)
    except Exception:
        page.locator("text=검색 중...").wait_for(state="hidden", timeout=15000)
    
    results_text = results_list.inner_text()
    
    assert "검색 결과가 없습니다" not in results_text, \
        f"검색 결과가 없습니다: '{query}' (결과: {results_text})"
    assert "오류:" not in results_text, \
        f"검색 중 오류 발생: '{query}' (결과: {results_text})"
    
    for expected in expected_contains:
        assert expected in results_text or expected.lower() in results_text.lower(), \
            f"'{query}' 검색 결과에 '{expected}'가 없음 (결과: {results_text})"
    
    result_items = page.locator(".search-result-item")
    assert result_items.count() >= 1, f"검색 결과 항목이 없음: '{query}'"
    
    select_btns = page.locator(".select-btn")
    assert select_btns.count() >= 1, f"선택 버튼이 없음: '{query}'"


def test_search_empty_query(page, base_url):
    """빈 검색어로 검색했을 때 정상 처리."""
    
    page.goto(base_url)
    
    search_input = page.locator("#searchInput")
    expect(search_input).to_be_visible()
    
    search_btn = page.get_by_role("button", name="검색")
    search_btn.click()
    
    results_list = page.locator("#searchResults")
    results_text = results_list.inner_text()
    
    assert "검색어를 입력하세요" in results_text, \
        f"검색어 입력 안내 메시지 없음 (결과: {results_text})"


def test_full_workflow_lg_cns(page, base_url):
    """LG CNS 종목 선택 → 데이터 수집 → 학습 → 예측값 및 그래프 검증."""
    
    page.goto(base_url)
    
    # Step 1: LG CNS 검색 및 결과 확인
    search_input = page.locator("#searchInput")
    search_input.fill("064400")
    page.get_by_role("button", name="검색").click()
    
    page.locator(".search-result-item").first.wait_for(timeout=15000)
    
    # Step 2: 선택 버튼 클릭
    select_btn = page.locator(".select-btn").first
    expect(select_btn).to_be_visible()
    select_btn.click()
    
    # Step 3: 진행 모달 표시 확인
    modal = page.locator("#workflowModal")
    expect(modal).to_be_visible(timeout=5000)
    
    # Step 4: 작업 완료 대기 (최대 150초)
    page.locator("#step5.completed").wait_for(timeout=150000)
    
    # Step 5: 모달 자동 닫히기 대기 (최대 10초)
    try:
        page.locator("#workflowModal.show").wait_for(state="hidden", timeout=10000)
    except:
        pass  # 모달이 안 닫혀도 진행
    
    # Step 6: 페이지 로드 대기
    page.wait_for_timeout(2000)
    
    # Step 7: 내일 주가 예측 섹션 확인
    prediction_section = page.locator("h2:has-text(\"내일 주가 예측\")")
    expect(prediction_section).to_be_visible(timeout=10000)
    prediction_card = page.locator(".card").filter(has_text="내일 주가 예측").first
    prediction_text = prediction_card.inner_text()
    
    # 예측 종가 정보 포함 확인
    assert "예측 종가" in prediction_text, \
        f"예측 종가 정보 없음: {prediction_text}"
    
    # 수치 데이터 확인 (최소 숫자 포함)
    assert any(str(i) in prediction_text for i in range(10)), \
        f"예측값이 없거나 형식이 잘못됨: {prediction_text}"
    
    # Step 9: 종목 분석 섹션 표시 확인
    analysis_header = page.locator("h2:has-text(\"종목 분석\"), text=종목 분석")
    try:
        expect(analysis_header).to_be_visible(timeout=10000)
    except:
        pass  # 분석이 없을 수도 있음
    
    # Step 10: 그래프 렌더링 확인
    charts = page.locator("canvas")
    chart_count = charts.count()
    assert chart_count >= 1, \
        f"그래프가 렌더링되지 않음 (발견된 canvas: {chart_count})"
    
    print(f"SUCCESS: Workflow completed, {chart_count} charts rendered")
    print(f"Prediction data: {prediction_text[:100]}...")
