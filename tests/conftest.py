"""E2E 테스트 설정 및 공통 fixture."""

import os
import sys
import time
import subprocess
import pytest
import requests
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def flask_server():
    """Flask 서버를 subprocess로 기동하고 health check."""
    test_port = 5001
    env = os.environ.copy()
    env["PORT"] = str(test_port)
    
    # Flask 서버 기동
    server_process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "app.py")],
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # health check: GET / 반복 (최대 30초)
    max_wait = 30
    start_time = time.time()
    base_url = f"http://localhost:{test_port}"
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(base_url, timeout=2)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.5)
    else:
        # timeout
        server_process.terminate()
        stdout, stderr = server_process.communicate(timeout=5)
        pytest.fail(f"Flask 서버 기동 실패 (timeout 30초). stderr: {stderr}")
    
    yield base_url
    
    # 정리
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture
def base_url(flask_server):
    """Flask 서버 base URL."""
    return flask_server
