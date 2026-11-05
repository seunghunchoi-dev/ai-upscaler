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
    # winget 으로 방금 설치하면 실행 중인 프로그램의 PATH 에는 아직 없다
    for base, sub in (("LOCALAPPDATA", "Microsoft/WinGet/Links"), ("ProgramFiles", "WinGet/Links")):
        link = Path(os.environ.get(base, "")) / sub / exe
        if os.environ.get(base) and link.exists():
            return str(link)
    raise FileNotFoundError(
        f"{name} 을(를) 찾을 수 없습니다. 프로그램 오른쪽 아래 [추가 구성요소] 에서 설치해 주세요."
    )


def ensure_dirs():
    for d in (CACHE, PRESETS):
        os.makedirs(d, exist_ok=True)
