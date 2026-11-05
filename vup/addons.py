"""설치 파일에 넣지 않는 추가 구성요소(FFmpeg, 얼굴 복원 모델)를 사용자가 공식 제공처에서 받는 기능.

둘 다 라이선스 때문에 함께 배포하지 않는다. 사용자가 화면에서 직접 고를 때만 받는다.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

from . import face as face_mod
from .paths import find_tool

GFPGAN_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx"
GFPGAN_SHA256 = "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385"
GFPGAN_LICENSE_URL = "https://github.com/TencentARC/GFPGAN/blob/master/LICENSE"
FFMPEG_LICENSE_URL = "https://ffmpeg.org/legal.html"
SETUP_GUIDE_URL = "https://github.com/seunghunchoi-dev/ai-upscaler/wiki/설치-및-실행"
_CHUNK = 1 << 20


def ffmpeg_ready():
    try:
        find_tool("ffmpeg")
        find_tool("ffprobe")
    except FileNotFoundError:
        return False
    return True


def face_ready():
    return face_mod.GFPGAN.exists()


def can_use_winget():
    return shutil.which("winget") is not None


def install_ffmpeg():
    """새 콘솔 창에서 winget 으로 FFmpeg 을 설치한다. 약관 동의는 사용자가 그 창에서 직접 한다."""
    r = subprocess.run(["winget", "install", "--id", "Gyan.FFmpeg", "--exact", "--source", "winget"],
                       creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
    return r.returncode == 0 or ffmpeg_ready()


def download_gfpgan(progress=None, cancelled=lambda: False, url=GFPGAN_URL, dest=None):
    """모델을 받아 해시를 확인한 뒤 제자리에 놓는다. 취소하면 False. progress(받은 바이트, 전체 바이트)."""
    dest = dest or face_mod.GFPGAN
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Upscaler"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(_CHUNK):
                if cancelled():
                    return False
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if digest.hexdigest() != GFPGAN_SHA256:
            raise ValueError("받은 파일이 공식 모델과 다릅니다. 다시 시도해 주세요.")
        os.replace(part, dest)
        return True
    finally:
        part.unlink(missing_ok=True)
