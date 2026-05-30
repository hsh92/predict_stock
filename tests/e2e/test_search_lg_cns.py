"""E2E 테스트: LG CNS 검색 기능."""

import pytest
from playwright.sync_api import expect


@pytest.mark.parametrize("query,expected_contains", [
    ("LG CNS", ["064400", "LG CNS"]),
    ("064400", ["064400"]),
])
def test_search_lg_cns(page, base_url, query, expected_contains):
    """LG CNS를 검색하여 결과가 정상 반환되는지 테스트."""
    
    # 페이지 로드
    page.goto(base_url)
    
    # 제목 확인 (페이지 정상 로드)
    assert "주가 예측" in page.title()
    
    # 검색 입력창 존재 확인
    search_input = page.locator("#searchInput")
    expect(search_input).to_be_visible()
    
    # 검색어 입력
    search_input.fill(query)
    
    # 검색 버튼 클릭
    search_btn = page.get_by_role("button", name="검색")
    search_btn.click()
    
    # 결과 목록이 나타날 때까지 대기 (최대 15초)
    # "검색 중..." 문구가 사라지고 결과가 표시될 때까지 대기
    results_list = page.locator("#searchResults")
    try:
        # 첫 번째 검색 결과 항목이 나타날 때까지 대기
        page.locator(".search-result-item").first.wait_for(timeout=15000)
    except Exception:
        # 혹은 "검색 중..." 문구가 사라질 때까지 대기
        page.locator("text=검색 중...").wait_for(state="hidden", timeout=15000)
    
    # 결과 텍스트 가져오기
    results_text = results_list.inner_text()
    
    # 검증 1: 오류나 "결과 없음" 메시지가 없어야 함
    assert "검색 결과가 없습니다" not in results_text, \
        f"검색 결과가 없습니다: '{query}' (결과: {results_text})"
    assert "오류:" not in results_text, \
        f"검색 중 오류 발생: '{query}' (결과: {results_text})"
    
    # 검증 2: 기대값이 결과에 포함되어야 함
    for expected in expected_contains:
        assert expected in results_text or expected.lower() in results_text.lower(), \
            f"'{query}' 검색 결과에 '{expected}'가 없음 (결과: {results_text})"
    
    # 검증 3: 검색 결과 항목 존재
    result_items = page.locator(".search-result-item")
    assert result_items.count() >= 1, \
        f"검색 결과 항목이 없음: '{query}'"
    
    # 검증 4: 각 항목에 선택 버튼 존재
    select_btns = page.locator(".select-btn")
    assert select_btns.count() >= 1, \
        f"선택 버튼이 없음: '{query}'"


def test_search_empty_query(page, base_url):
    """빈 검색어로 검색했을 때 정상 처리."""
    
    page.goto(base_url)
    
    # 검색 입력창 존재
    search_input = page.locator("#searchInput")
    expect(search_input).to_be_visible()
    
    # 빈 상태로 검색 버튼 클릭
    search_btn = page.get_by_role("button", name="검색")
    search_btn.click()
    
    # 결과 목록 확인
    results_list = page.locator("#searchResults")
    results_text = results_list.inner_text()
    
    # 검색어 입력 안내 메시지 표시되어야 함
    assert "검색어를 입력하세요" in results_text, \
        f"검색어 입력 안내 메시지 없음 (결과: {results_text})"
