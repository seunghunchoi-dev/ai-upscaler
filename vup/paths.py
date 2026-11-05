"""프로그램 폴더 기준 경로. 절대경로를 쓰지 않으므로 폴더째 옮겨도 동작한다."""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BIN = ROOT / "bin"
CACHE = ROOT / "cache"
PRESETS = ROOT / "presets"
CONFIG = ROOT / "config.json"


def find_tool(name):
    """bin 폴더의 실행파일을 우선 사용하고, 없으면 PATH 에서 찾는다."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    local = BIN / exe
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} 을(를) 찾을 수 없습니다. bin 폴더에 {exe} 를 넣어 주세요."
    )


def ensure_dirs():
    for d in (CACHE, PRESETS):
        os.makedirs(d, exist_ok=True)
