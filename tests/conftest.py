"""pytest 공통 설정
src/trendops/ 패키지를 pytest가 인식할 수 있도록 경로 설정
"""
import sys
from pathlib import Path

# src/ 디렉토리를 Python path에 추가
# 이렇게 해야 from trendops.xxx import yyy 가 동작
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))