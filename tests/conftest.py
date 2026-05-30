"""E2E 테스트 설정 및 공통 fixture."""

import os
import sys
import time
import socket
import subprocess
import pytest
import requests
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent


def find_free_port():
    """사용 가능한 포트를 찾습니다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(scope="session")
def flask_server():
    """Flask 서버를 subprocess로 기동하고 health check."""
    test_port = find_free_port()
    env = os.environ.copy()
    env["PORT"] = str(test_port)
    env["FLASK_ENV"] = "test"
    
    # Flask 서버 기동
    server_process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "app.py")],
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    
    # health check: GET / 반복 (최대 45초)
    max_wait = 45
    start_time = time.time()
    base_url = f"http://localhost:{test_port}"
    
    last_error = None
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(base_url, timeout=2)
            if response.status_code == 200:
                break
        except requests.RequestException as e:
            last_error = e
        time.sleep(1)
    else:
        # timeout
        try:
            server_process.terminate()
            server_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
        error_msg = f"Flask 서버 기동 실패 (timeout {max_wait}초, 포트: {test_port})"
        if last_error:
            error_msg += f". 마지막 오류: {last_error}"
        pytest.fail(error_msg)
    
    yield base_url
    
    # 정리
    try:
        server_process.terminate()
        try:
            server_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
    except Exception:
        pass


@pytest.fixture(scope="session")
def base_url(flask_server):
    """Flask 서버 base URL."""
    return flask_server
