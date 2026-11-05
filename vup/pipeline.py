"""전체 처리 흐름.

  [읽기] → [보간(RIFE)] → [AI 복원·업스케일] → [후처리 필터] → [인코딩]
  각 단계가 별도 스레드에서 동시에 돌아가며 큐로 연결된다.
  긴 영상은 조각(chunk) 단위로 저장해 중간에 멈춰도 이어서 처리할 수 있다.
"""
import hashlib
import json
import logging
import math
import queue
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import backends, esrgan, filters, gpufilter, video_io
from . import face as face_mod
from .paths import CACHE, MODELS

log = logging.getLogger(__name__)


@dataclass
class Settings:
    # 해상도
    target_height: int = 2160     # 출력 해상도 (짧은 변 기준: 480/720/1080/1440/2160/4320). 0 이면 scale 사용
    scale: float = 1.0            # target_height 가 0 일 때 출력 배율 (원본 대비)
    proc_scale: float = 0.75      # AI 처리 해상도 (원본 대비). 낮을수록 빠름
    ai_restore: bool = True       # AI 복원·업스케일 사용
    sr_model: str = "general_v3"  # general_v3(빠름) | x4plus | x2plus (고품질·느림)
    sr_int8: bool = False         # AI 복원 경량화 (INT8, general_v3 전용 · 약 1.45배 빠름 · 미세 노이즈 약간)
    denoise: float = 0.6          # 압축 노이즈 제거 강도 0~1 (general_v3 전용)
    face_enhance: bool = True     # 얼굴 복원 (GFPGAN)
    face_strength: float = 0.6    # 얼굴 복원 강도 0~1
    face_min: int = 32            # 이보다 작은 얼굴(업스케일 후 픽셀)은 복원하지 않음
    face_max: int = 0             # 한 화면에서 복원할 최대 얼굴 수 (0 = 제한 없음, 큰 얼굴부터)
    # 프레임
    fps: str = "60"               # "src" | "x2" | "x3" | "x4" | 숫자(예: "60", "59.94")
    rife_model: str = "rife_v4.26"
    scene_threshold: float = 0.15  # 장면 전환 감지 민감도 (낮을수록 민감)
    dedup: bool = True            # 중복 프레임 보정 (움직임 중 끊김 방지)
    interp_res: str = "auto"      # 보간 해상도: auto(작은 얼굴 구간만 원본) | fast(작은 얼굴 주변만 원본) |
                                  #              full(원본) | proc(처리 해상도, 빠름)
    # 스타일 (기본값 = '깨끗하게')
    texture: float = 0.05         # 원본 질감 유지
    soft_skin: float = 0.2        # 피부 보정 (뽀샤시)
    glow: float = 0.0             # 빛 번짐 (뽀샤시)
    brighten: float = 0.0         # 화사함
    sharpen: float = 0.2          # 선명도
    grain: float = 0.0            # 필름 그레인
    clarity: float = 0.0          # 명료도 (입체감)
    vibrance: float = 0.0         # 생생함 (채도 낮은 색 위주로 채도 올림)
    engine: str = "ai"            # ai | filter (AI 없이 필터만 — 빠름, 원본 프레임레이트)
    # 기타
    deinterlace: str = "auto"     # auto | on | off
    device_sr: str = "auto"
    device_rife: str = "auto"
    device_face: str = "auto"     # 얼굴 복원 장치 (auto: 업스케일과 같은 장치)
    codec: str = "h265"           # h264 | h265 | av1
    encoder: str = "auto"         # auto(하드웨어 우선) | software | nvenc | qsv | amf
    crf: int = 20
    ten_bit: bool = True
    audio: str = "auto"           # auto(가능하면 원본 그대로) | aac | none
    chunk_seconds: int = 30

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self):
        return asdict(self)


QUALITY_PRESETS = {
    # AI 처리의 화질·속도 프리셋. AI 없이 필터만 쓰는 방식은 '복원' 탭의 처리 방식에서 고른다 (FILTER_LOOK).
    # 이름은 '용도 + 결과' 로 (목록 순서 = 화면 순서: 음악방송 추천 → 빠른 복원 → 일반)
    #
    # Core Ultra 5 125H (내장 GPU) + 32GB 실측 기반. 4K 60fps · 경량화 · 처리 50% · 큰 얼굴 2개.
    # 처리 33% 일 때 2.25fps 였던 구간이 GPU 후처리(gpufilter)로 3.61fps 가 되어, 아낀 시간을 화질에 다시 씀:
    # 처리 50% 로 같은 구간 2.12fps (33% 이전과 거의 같은 시간). 마이크 질감·귀걸이·머리카락·옷깃이 확실히
    # 또렷하고, 33% 에서 뭉개지던 단체 샷의 작은 얼굴도 덜 뭉개진다. (40% 는 2.90fps)
    # 원본 32px 미만 얼굴 제외(복원 효과가 미미), 보간 '빠른 자동'(작은 얼굴 주변만 원본 해상도 —
    # 원본 해상도 전체 보간이 메모리를 나눠 쓰는 CPU 의 병목이었음, 단체 샷 +63%, 깜빡임 동일)
    "음악방송 4K 60fps (추천)": dict(sr_model="general_v3", proc_scale=0.5, rife_model="rife_v4.26",
                                 target_height=2160, fps="60", face_enhance=True, face_max=2, face_min=64,
                                 interp_res="fast", sr_int8=True),
    # 위와 같은 화질로 1080p · 원본 프레임레이트(보간 없음). 출력 프레임이 절반이라 약 2배 빠름
    # (3분 영상 약 35분 — 4K 60fps 실측에서 추정). 보간을 안 하므로 interp_res 는 기본값 —
    # 위 프리셋과 값이 겹쳐 화면의 프리셋 표시가 헷갈리지 않도록 구분하는 역할도 한다.
    "음악방송 1080p (보간 없음)": dict(sr_model="general_v3", proc_scale=0.5, rife_model="rife_v4.26",
                                  target_height=1080, fps="src", face_enhance=True, face_max=2, face_min=64,
                                  interp_res="auto", sr_int8=True),
    # 1440p · 처리 50% · 얼굴 3개, 경량화 없음. 실측 단체 샷 1.14 / 클로즈업 1.75fps
    # (출력 1440p 는 4K 대비 +12%, 큰 얼굴 3개까지 +3~6%)
    "음악방송 1440p 60fps (얼굴 3명)": dict(sr_model="general_v3", proc_scale=0.5, rife_model="rife_v4.26",
                                      target_height=1440, fps="60", face_enhance=True, face_max=3,
                                      face_min=64, interp_res="fast"),
    # AI 복원만 빠르게: 원본 해상도 유지, 보간·얼굴 복원 없음, INT8 경량 모델(약 1.45배 빠름, 미세 노이즈 약간).
    # 실측 1080p 30fps 5.3~5.4fps (3분 약 17분)
    "빠른 복원 (원본 크기)": dict(sr_model="general_v3", proc_scale=0.5, rife_model="rife_v4.26",
                             sr_int8=True, target_height=0, scale=1.0, fps="src", face_enhance=False),
    # 경량화 + 처리 해상도 33%. 실측 11.6~11.8fps (3분 약 8분, 4분 약 10분).
    # 중간 샷·클로즈업은 50% 와 비슷하지만 단체 샷의 작은 얼굴(원본 40px 안팎)은 눈에 띄게 더 뭉개진다
    "가장 빠른 복원 (원본 크기)": dict(sr_model="general_v3", proc_scale=0.33, rife_model="rife_v4.26",
                                 sr_int8=True, target_height=0, scale=1.0, fps="src", face_enhance=False),
    # 일반: 처리 해상도만 정하고 출력·얼굴 설정은 그대로 둔다.
    # 보간(RIFE)은 CPU 에서 업스케일과 동시에 돌고 4.26 도 lite 와 속도가 거의 같아서 모두 4.26 사용.
    # 옷·배경 디테일은 처리 해상도가 가장 크게 좌우한다: General v3 100% 가 x4plus 50% 보다
    # 2.5~4배 빠르면서 형태(무늬·잔가지)가 더 정확했다. 최고 화질은 원본 해상도를 RRDB 로 처리 (매우 느림).
    "일반 · 빠름": dict(sr_model="general_v3", proc_scale=0.5, rife_model="rife_v4.26"),
    "일반 · 균형": dict(sr_model="general_v3", proc_scale=0.75, rife_model="rife_v4.26"),
    "일반 · 고화질": dict(sr_model="general_v3", proc_scale=1.0, rife_model="rife_v4.26"),
    "일반 · 최고 화질 (매우 느림)": dict(sr_model="x2plus", proc_scale=1.0, rife_model="rife_v4.26"),
}

# 처리 방식을 '필터만'(AI 없이 FFmpeg 필터: 블록 완화·노이즈 제거 → 명료도 → 밝기 → 확대)으로 바꿀 때 넣는 값.
# 없던 디테일은 만들지 못하지만 실시간에 가깝게 빠름. 보간 없음(원본 프레임레이트).
# '일반 · 최고 화질' 결과(실제 음악방송 5장면)에 가깝게 자동 탐색한 뒤 더 밝고 깨끗하게 조정:
# 생생함은 채도가 AI 결과보다 크게 높아져서(+17) 끄고, 원본 크기 CAS 선명화는 AI 결과와 모양이 달라져서 끔
FILTER_LOOK = dict(denoise=0.6, sharpen=0.0, clarity=0.6, vibrance=0.0, brighten=0.35)

STYLE_PRESETS = {
    "자연스럽게": dict(texture=0.15, soft_skin=0.0, glow=0.0, brighten=0.0, sharpen=0.1, grain=0.15),
    "깨끗하게": dict(texture=0.05, soft_skin=0.2, glow=0.0, brighten=0.0, sharpen=0.2, grain=0.0),
    "뽀샤시": dict(texture=0.0, soft_skin=0.6, glow=0.35, brighten=0.3, sharpen=0.0, grain=0.0),
    "선명하게": dict(texture=0.15, soft_skin=0.0, glow=0.0, brighten=0.0, sharpen=0.5, grain=0.1),
}

RIFE_MODELS = {"rife_v4.26": "RIFE 4.26 (고품질)", "rife_v4.22_lite": "RIFE 4.22 lite (빠름)"}

RESOLUTIONS = [  # (이름, 짧은 변 픽셀)
    ("SD 480p", 480), ("HD 720p", 720), ("FHD 1080p", 1080), ("QHD 1440p (2K)", 1440),
    ("UHD 2160p (4K)", 2160), ("8K 4320p", 4320),
]


def resolution_name(short_side):
    return next((n for n, h in RESOLUTIONS if h == short_side), f"{short_side}p")


def _even(x):
    return max(2, int(round(x / 2)) * 2)


def output_factor(info, s):
    """원본 대비 출력 배율. 가로·세로 영상 모두 짧은 변 기준(유튜브와 같은 방식)."""
    if s.target_height:
        return s.target_height / min(info.width, info.height)
    return s.scale


def target_fps(src, mode):
    mode = str(mode).strip().lower()
    if mode in ("src", "", "원본"):
        return src
    if mode.startswith("x"):
        return src * int(mode[1:])
    f = Fraction(mode).limit_denominator(1001)
    # 29.97 같은 NTSC 계열 원본이면 60 → 59.94 로 맞춰 프레임 간격을 고르게 한다
    if src.denominator == 1001 and f.denominator == 1:
        f = f * 1000 / 1001
    return f


class Plan:
    """영상 정보와 설정으로 처리 크기·프레임 수를 계산한다."""

    def __init__(self, info, s):
        self.info, self.s = info, s
        self.factor = output_factor(info, s)
        ps = s.proc_scale if s.ai_restore else max(s.proc_scale, min(1.0, self.factor))
        self.pw, self.ph = _even(info.width * ps), _even(info.height * ps)
        # 얼굴 복원을 쓰면 원본 해상도로 읽고 보간한다. AI 업스케일에는 줄여서 넣고,
        # 얼굴은 원본 해상도에서 찾아 복원 재료로 쓴다 (작은 얼굴도 살리면서 업스케일은 빠르게)
        self.faces = s.ai_restore and s.face_enhance and s.face_strength > 0 and face_mod.available()
        self.rw, self.rh = (_even(info.width), _even(info.height)) if self.faces else (self.pw, self.ph)
        self.ow, self.oh = _even(info.width * self.factor), _even(info.height * self.factor)
        if s.target_height:
            # 854x480 처럼 거의 16:9 인 원본은 1920x1080 같은 표준 크기로 맞춘다
            landscape = info.width >= info.height
            long_side = max(info.width, info.height) / min(info.width, info.height)
            for ratio in (16 / 9, 4 / 3, 1.0):
                if abs(long_side / ratio - 1) < 0.01:
                    other = _even(s.target_height * ratio)
                    self.ow, self.oh = (other, s.target_height) if landscape else (s.target_height, other)
                    break
        # AI 는 처리 해상도의 모델 배율(4배 또는 2배)까지 만든다. 그보다 크게 출력하면 나머지는 단순 확대
        self.ai_scale = esrgan.model_scale(s.sr_model) if s.sr_model in esrgan.SR_MODELS else 4
        self.ai_short = s.ai_restore and self.factor > ps * self.ai_scale + 1e-6
        self.fin = info.fps
        self.fout = target_fps(info.fps, s.fps)
        self.ratio = self.fin / self.fout             # 출력 1프레임당 원본 프레임 수
        self.interp = self.fout != self.fin
        self.deint = s.deinterlace == "on" or (s.deinterlace == "auto" and info.interlaced)

    def out_frames(self, n_src):
        return math.ceil(n_src / self.ratio)


# ── 처리기 ────────────────────────────────────────────────────────────────

_runner_cache = {}


_runner_lock = threading.Lock()


def _cached_runner(key, make):
    # 얼굴 복원 보조 장치(NPU)는 처리 중에 다른 스레드에서 준비하므로 잠금. 준비(컴파일)는 잠금 밖에서
    with _runner_lock:
        if key in _runner_cache:
            return _runner_cache[key]
    obj = make()
    with _runner_lock:
        if key not in _runner_cache:
            if len(_runner_cache) >= 6:
                _runner_cache.pop(next(iter(_runner_cache)))
            _runner_cache[key] = obj
        return _runner_cache[key]


def resolve_devices(s):
    sr = s.device_sr if s.device_sr != "auto" else backends.auto_device()
    rife = s.device_rife
    if rife == "auto":
        # 내장 GPU 는 업스케일에 쓰고 보간은 CPU 에 맡기면 둘이 동시에 일한다
        ids = [d for d, _ in backends.list_devices()]
        # (AI 화질 복원을 꺼서 GPU 가 비어 있어도 원본 해상도 보간은 CPU 가 더 빨랐다: 1.10 vs 1.23초/장)
        if sr.startswith("ov:GPU") and "ov:CPU" in ids and "외장" not in backends.device_label(sr):
            rife = "ov:CPU"
        elif sr == "ov:NPU":
            rife = "ov:CPU" if "ov:CPU" in ids else ids[-1]
        elif sr == "ort:trt":
            # 보간은 해상도가 장면마다 바뀌어(얼굴 주변 조각 등) TensorRT 엔진을 계속 새로 만들게 되므로 CUDA 로
            rife = "ort:cuda"
        else:
            rife = sr
    return sr, rife


def resolve_face_device(s, sr):
    return sr if s.device_face == "auto" else s.device_face


def interp_devices(s, rife_dev):
    """보간(RIFE)을 나눠 맡을 장치들. AI 화질 복원을 안 하면(보간만) GPU 가 비어 있으므로
    CPU 와 GPU 가 프레임을 나눠 보간한다. 실측(1080p 한 장): CPU 612ms · GPU 882ms → 합치면 약 1.7배.
    NPU 는 1080p 보간에서 장치가 멈춰(DEVICE_LOST) 쓰지 않는다."""
    devs = [rife_dev]
    if not s.ai_restore and s.device_rife == "auto" and rife_dev == "ov:CPU":
        gpu = next((d for d, _ in backends.list_devices() if d.startswith("ov:GPU")), None)
        if gpu:
            devs.append(gpu)
    return devs


def extra_face_device(face_dev):
    """얼굴 복원을 함께 나눠 맡을 두 번째 장치 (없으면 None).
    Intel GPU 는 업스케일·얼굴 복원·후처리로 꽉 차고 NPU 는 놀고 있어서, NPU 가 프레임을 나눠 복원한다.
    실측(GFPGAN 한 얼굴): GPU 133ms · NPU 300ms, 두 장치 결과 차이 평균 0.2/255 (깜빡임 없음)."""
    if face_dev.startswith("ov:GPU") and "ov:NPU" in dict(backends.list_devices()):
        return "ov:NPU"
    return None


class Upscaler:
    def __init__(self, plan, device, track_faces=True):
        self.plan = plan
        s = plan.s
        if s.ai_restore:
            model = s.sr_model if s.sr_model in esrgan.available_models() else "general_v3"
            # denoise 는 general_v3 에서만 그래프를 바꾸므로 다른 모델은 키에서 뺀다
            dn = round(s.denoise, 3) if model == "general_v3" else None
            int8 = esrgan.int8_path(s.denoise) if (s.sr_int8 and model == "general_v3"
                                                   and device.startswith("ov:")) else None
            shape = (1, plan.ph, plan.pw, 3)
            if int8 is not None:
                # 미리 만들어 둔 INT8 경량 모델 (가장 가까운 노이즈 제거 단계)
                self.run = _cached_runner(("sr-int8", str(int8), device, plan.pw, plan.ph),
                                          lambda: backends.Runner(str(int8), device, shape))
            else:
                key = ("sr", model, dn, device, plan.pw, plan.ph)
                self.run = _cached_runner(key, lambda: backends.Runner(
                    esrgan.build_onnx(s.denoise, model), device, shape))
        self.face = None
        self.face_extra = None  # 얼굴 복원을 나눠 맡을 두 번째 장치 (process 에서 따로 준비)
        if plan.faces:
            fdev = resolve_face_device(s, device)
            self.face_extra = extra_face_device(fdev)
            self.face = _cached_runner(("face", fdev), lambda: face_mod.FaceEnhancer(fdev))
            self.face.strength, self.face.min_size = s.face_strength, s.face_min
            self.face.max_faces = s.face_max
            self.face.reset()
        self.track_faces = track_faces

    def infer(self, frame):
        """AI 업스케일만 (GPU 스레드용).
        frame: 읽기 해상도(rw, rh) uint8 → float32 (배율×ph, 배율×pw, 3) 또는 None."""
        p = self.plan
        if not p.s.ai_restore:
            return None
        proc = frame
        if frame.shape[1] != p.pw or frame.shape[0] != p.ph:
            proc = cv2.resize(frame, (p.pw, p.ph), interpolation=cv2.INTER_AREA)
        return self.run(np.ascontiguousarray(proc)[None])[0]

    def enhance_faces(self, y, frame, dets=None):
        """얼굴 복원 (별도 스레드에서 프레임 순서대로 호출됨 — 얼굴 추적에 순서가 필요).
        얼굴은 읽기 해상도(원본) 프레임에서 찾아 복원 재료로 쓴다. dets 는 보간 단계에서 미리 찾은 얼굴."""
        if self.face is None or y is None:
            return y
        return self.face(y, track=self.track_faces, src=frame, dets=dets)

    def plan_faces(self, y, frame, dets=None):
        """얼굴 추적 → 복원할 얼굴 목록 (프레임 순서대로 한 스레드에서 호출)."""
        if self.face is None or y is None:
            return []
        return self.face.plan(y, src=frame, dets=dets, track=self.track_faces)

    def restore_faces(self, y, frame, plans, run=None):
        """plan_faces 결과대로 복원 (여러 스레드·장치에서 동시에 호출 가능). run: GFPGAN Runner (없으면 기본)."""
        if self.face is None or y is None or not plans:
            return y
        return self.face.apply(y, plans, src=frame, run=run)

    def finish(self, y, frame):
        """크기 맞추기와 원본 확대 (CPU 후처리 스레드용) → (결과, 원본 단순확대) float32 (oh, ow, 3)."""
        p = self.plan
        base = None
        if p.s.texture > 0 or y is None:
            base = cv2.resize(frame, (p.ow, p.oh), interpolation=cv2.INTER_CUBIC)
            base = cv2.multiply(base, 1 / 255, dtype=cv2.CV_32F)
        if y is None:
            return base, None
        if y.shape[1] != p.ow or y.shape[0] != p.oh:
            interp = cv2.INTER_AREA if y.shape[1] > p.ow else cv2.INTER_CUBIC
            y = cv2.resize(y, (p.ow, p.oh), interpolation=interp)
        return y, base

    def __call__(self, frame):
        return self.finish(self.enhance_faces(self.infer(frame), frame), frame)


class _RifeAt:
    """한 해상도로 컴파일한 RIFE."""

    def __init__(self, plan, device, w, h):
        self.w, self.h = w, h
        self.pad_h = (32 - h % 32) % 32
        self.pad_w = (32 - w % 32) % 32
        ph, pw = h + self.pad_h, w + self.pad_w
        model = MODELS / f"{plan.s.rife_model}.onnx"
        self.run = _cached_runner(("rife", plan.s.rife_model, device, pw, ph),
                                  lambda: backends.Runner(model, device, (1, 7, ph, pw)))


class Interpolator:
    """읽기 해상도(rw, rh) 프레임 사이를 보간한다.

    얼굴 복원을 쓰면 읽기 해상도가 원본(1080p)이라 보간이 CPU 의 가장 큰 병목이 된다.
    AI 업스케일 입력은 어차피 처리 해상도(pw, ph)이므로, 처리 해상도에서 보간해도 결과가 같다.
    다르게 되는 것은 '작은 얼굴의 복원 재료'뿐이라(중간 프레임만 흐려져 깜빡임),
    interp_res='auto' 는 작은 얼굴이 있는 구간만 원본 해상도로, 나머지는 처리 해상도로 보간한다."""

    CROP = 256  # 'fast' 모드: 작은 얼굴 주변만 원본 해상도로 보간하는 조각 크기

    def __init__(self, plan, device):
        self.plan = plan
        self.out_w, self.out_h = plan.rw, plan.rh
        mode = plan.s.interp_res
        reduced = (plan.pw, plan.ph) != (plan.rw, plan.rh)
        self.full = _RifeAt(plan, device, plan.rw, plan.rh) if mode not in ("proc", "fast") else None
        self.low = _RifeAt(plan, device, plan.pw, plan.ph) if reduced and mode != "full" else None
        if self.full is None and self.low is None:
            self.full = _RifeAt(plan, device, plan.rw, plan.rh)
        # 'fast': 화면 전체는 처리 해상도로 보간하고, 작은 얼굴 주변(256x256)만 원본 해상도로 다시 보간한다.
        # 원본 해상도 전체 보간은 내장 GPU 와 메모리를 나눠 쓰는 CPU 에서 가장 큰 병목이었다.
        self.crop = None
        if mode == "fast" and reduced and plan.rw >= self.CROP and plan.rh >= self.CROP:
            self.crop = _RifeAt(plan, device, self.CROP, self.CROP)
            m = np.zeros((self.CROP, self.CROP), np.float32)
            cv2.rectangle(m, (24, 24), (self.CROP - 25, self.CROP - 25), 1.0, -1)
            self.crop_mask = cv2.GaussianBlur(m, (0, 0), 8)[..., None]  # 경계를 부드럽게 섞는 마스크

    @staticmethod
    def _prep(r, f):
        if f.shape[1] != r.w or f.shape[0] != r.h:
            f = cv2.resize(f, (r.w, r.h), interpolation=cv2.INTER_AREA)
        x = f.astype(np.float32) * (1 / 255)
        if r.pad_h or r.pad_w:
            x = cv2.copyMakeBorder(x, 0, r.pad_h, 0, r.pad_w, cv2.BORDER_REPLICATE)
        return x.transpose(2, 0, 1)

    def __call__(self, a, b, t, low=False):
        r = self.low if (low and self.low is not None) or self.full is None else self.full
        out = self._infer(r, a, b, t)
        if (r.w, r.h) != (self.out_w, self.out_h):
            out = cv2.resize(out, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)
        return out

    def refine_faces(self, out, a, b, t, centers):
        """'fast' 모드: out(처리 해상도 보간을 키운 것)의 작은 얼굴 주변만 원본 해상도 보간으로 바꾼다.
        centers: 얼굴 중심 좌표들 (읽기 해상도 기준)."""
        if self.crop is None or not centers:
            return out
        C, h, w = self.CROP, self.out_h, self.out_w
        for cx, cy in centers:
            x0 = int(min(max(0, cx - C / 2), w - C))
            y0 = int(min(max(0, cy - C / 2), h - C))
            patch = self._infer(self.crop, a[y0:y0 + C, x0:x0 + C], b[y0:y0 + C, x0:x0 + C], t)
            roi = out[y0:y0 + C, x0:x0 + C].astype(np.float32)
            out[y0:y0 + C, x0:x0 + C] = (roi + (patch.astype(np.float32) - roi) * self.crop_mask + 0.5).astype(np.uint8)
        return out

    def _infer(self, r, a, b, t):
        """r(한 해상도로 준비한 RIFE)로 a·b 사이 t 시점 프레임을 만든다 (r 해상도 그대로 uint8)."""
        xa, xb = self._prep(r, a), self._prep(r, b)
        # 조명 깜빡임처럼 두 프레임의 밝기가 크게 다르면 RIFE 가 움직임을 못 찾고 두 장을 섞어 버린다.
        # → b 의 밝기·대비를 a 에 맞춰 움직임을 계산하고, 결과 밝기는 시간 비율(t)만큼 b 쪽으로 옮긴다.
        ma, sa = xa.mean(axis=(1, 2), keepdims=True), xa.std(axis=(1, 2), keepdims=True) + 1e-4
        mb, sb = xb.mean(axis=(1, 2), keepdims=True), xb.std(axis=(1, 2), keepdims=True) + 1e-4
        relight = float(np.abs(ma - mb).max()) > 0.03 or float(np.abs(np.log(sb / sa)).max()) > 0.1
        if relight:
            xb = np.clip((xb - mb) / sb * sa + ma, 0, 1)
        x = np.concatenate([xa, xb, np.full((1,) + xa.shape[1:], t, np.float32)])[None]
        y = r.run(x)[0, :, :r.h, :r.w]
        if relight:
            mt, st = ma + (mb - ma) * t, sa + (sb - sa) * t
            y = (y - ma) / sa * st + mt
        return (np.clip(y.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)


def _thumb(f):
    return cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255)


def frame_diff(a, b):
    """두 프레임 축소본의 움직임 차이 (0~1). 중복 프레임 판단용.
    평균 밝기를 빼고 비교하므로 조명 깜빡임은 움직임으로 치지 않는다."""
    ta, tb = _thumb(a), _thumb(b)
    ta -= ta.mean(axis=(0, 1))
    tb -= tb.mean(axis=(0, 1))
    return float(np.abs(ta - tb).mean())


def brightness_jump(a, b):
    """두 프레임의 평균 밝기 차이 (0~1)."""
    return float(abs(_thumb(a).mean() - _thumb(b).mean()))


_ECC_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)


def aligned_similarity(a, b):
    """카메라 줌·이동과 밝기 변화를 보정해 겹친 뒤의 유사도 (상관계수, 1 = 같은 장면).

    빠른 줌이나 조명 깜빡임은 화면이 크게 바뀌어도 보정하면 거의 같으므로 높게 나오고,
    다른 카메라로 바뀐 컷은 어떻게 겹쳐도 맞지 않으므로 낮게 나온다."""
    ga = cv2.cvtColor(cv2.resize(a, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(cv2.resize(b, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    sa, sb = float(ga.std()), float(gb.std())
    if sa < 3 or sb < 3:  # 거의 단색 화면(암전 등): 밝기 차이로만 판단
        return 1.0 if (sa < 3) == (sb < 3) and abs(float(ga.mean() - gb.mean())) < 30 else 0.0
    na, nb = (ga - ga.mean()) / sa, (gb - gb.mean()) / sb  # 밝기·대비 정규화
    corr = float((na * nb).mean())
    if corr > 0.9:
        return corr
    warp = np.eye(2, 3, dtype=np.float32)
    try:  # 거친 단계 → 세밀한 단계로 줌·이동(어파인) 추정
        small_a = cv2.resize(na, (64, 36), interpolation=cv2.INTER_AREA)
        small_b = cv2.resize(nb, (64, 36), interpolation=cv2.INTER_AREA)
        _, warp = cv2.findTransformECC(small_a, small_b, warp, cv2.MOTION_AFFINE, _ECC_CRIT, None, 3)
        warp[:, 2] *= 2
        cc, _ = cv2.findTransformECC(na, nb, warp, cv2.MOTION_AFFINE, _ECC_CRIT, None, 3)
        return max(corr, float(cc))
    except cv2.error:  # 맞출 수 없음 = 다른 장면일 가능성이 큼
        return corr


def _chroma_hist(f):
    ycc = cv2.cvtColor(cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2YCrCb)
    h = cv2.calcHist([ycc], [1, 2], None, [32, 32], [0, 256, 0, 256])
    # 히스토그램을 번지게 해서, 색이 조금 바뀐 것(원색 LED 조명 등)은 작게, 다른 장면은 크게 나오도록
    h = cv2.GaussianBlur(h, (0, 0), 2.5)
    return cv2.normalize(h, h, 1, 0, cv2.NORM_L1)


def color_change(a, b):
    """색 분포 차이 (0 = 같음, 1 = 완전히 다름). 밝기는 제외하므로 조명 깜빡임에는 둔하다."""
    return float(cv2.compareHist(_chroma_hist(a), _chroma_hist(b), cv2.HISTCMP_BHATTACHARYYA))


def is_scene_cut(a, b, threshold):
    """장면 전환(카메라 컷) 판단. threshold 가 낮을수록 민감 (1 이상이면 끔).

    줌·이동·밝기 변화를 보정해도 모양이 안 맞거나, 색 분포가 크게 바뀌면 컷으로 본다.
    (확대된 하늘처럼 모양만으로 구분이 안 되는 장면은 색으로 잡는다)"""
    if threshold >= 1:
        return False
    if color_change(a, b) > 0.3 + 2.0 * threshold:
        return True
    return aligned_similarity(a, b) < 1.0 - 3.0 * threshold


def is_duplicate(d, last_motion):
    """움직이던 중(last_motion)에 거의 차이 없는 프레임(d)이 나오면 중복 프레임으로 본다.
    정지 장면은 원래 차이가 없으므로 중복으로 보지 않는다."""
    return last_motion > 0.004 and d < max(0.0015, 0.15 * last_motion)


# ── 미리보기 ───────────────────────────────────────────────────────────────

def preview_frame(path, s, time_sec):
    """한 장면을 처리해 (원본 확대, 결과) uint8 이미지를 돌려준다."""
    info = video_io.probe(path)
    plan = Plan(info, s)
    if s.engine == "filter":
        from . import fastfilter
        return fastfilter.preview(path, s, info, plan, min(max(0, time_sec), max(0, info.duration - 0.1)))
    sr_dev, _ = resolve_devices(s)
    idx = int(min(max(0, time_sec), max(0, info.duration - 0.1)) * info.fps)
    orig = next(iter(video_io.FrameReader(info, info.width, info.height, plan.deint, idx, 1)))
    # 본 처리와 똑같이 읽기 해상도 프레임을 넘긴다 (얼굴 복원 시 원본 해상도, 아니면 처리 해상도)
    frame = cv2.resize(orig, (plan.rw, plan.rh), interpolation=cv2.INTER_AREA) \
        if (plan.rw, plan.rh) != (info.width, info.height) else orig
    sr, base = Upscaler(plan, sr_dev, track_faces=False)(frame)
    before = cv2.resize(orig, (plan.ow, plan.oh), interpolation=cv2.INTER_CUBIC)
    return before, filters.apply_all(sr, base, s)


# ── 본 처리 ────────────────────────────────────────────────────────────────

class Stopped(Exception):
    pass


@dataclass
class Progress:
    done: int
    total: int
    fps: float
    eta: float
    stage: str
    frame: object = None   # 최근 출력 프레임 (미리보기용, 가끔만 채워짐)


def job_dir_for(path, s, start, duration):
    p = Path(path)
    st = p.stat()
    key = json.dumps([str(p.resolve()), st.st_size, int(st.st_mtime), s.to_dict(), start, duration],
                     sort_keys=True, ensure_ascii=False)
    return CACHE / "jobs" / hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def process(path, out_path, s, on_progress=None, stop_event=None, start=0.0, duration=None):
    """영상을 처리해 out_path 에 저장한다. duration 을 주면 그 구간만(샘플) 처리한다."""
    stop_event = stop_event or threading.Event()
    on_progress = on_progress or (lambda p: None)
    info = video_io.probe(path)
    if s.engine == "filter":
        # AI 없이 필터만: FFmpeg 안에서 전부 처리 (보간 없음 → 원본 프레임레이트)
        from . import fastfilter
        from dataclasses import replace
        s = replace(s, fps="src")
        plan = Plan(info, s)
        encoder = video_io.pick_encoder(s.codec, s.encoder)
        on_progress(Progress(0, 1, 0, 0, "필터 처리 준비 중"))
        return fastfilter.process(path, out_path, s, info, plan, encoder, on_progress, stop_event,
                                  start, duration, Progress)
    plan = Plan(info, s)
    sr_dev, rife_dev = resolve_devices(s)
    encoder = video_io.pick_encoder(s.codec, s.encoder)
    audio_src = path if (s.audio != "none" and info.audio_codec) else None

    start_src = int(round(start * info.fps))
    n_src = info.frames - start_src
    if duration:
        n_src = min(n_src, int(round(duration * info.fps)))
    n_src = max(1, n_src)
    total_out = plan.out_frames(n_src)

    on_progress(Progress(0, total_out, 0, 0, "모델 준비 중"))
    upscaler = Upscaler(plan, sr_dev)
    interp = Interpolator(plan, rife_dev) if plan.interp else None
    # 보간을 여러 장치가 나눠 맡으면 프레임 순서가 섞이므로, 순서가 필요한 얼굴 추적이 없을 때만
    rife_devs = interp_devices(s, rife_dev) if interp is not None and not plan.faces else []
    interps = [interp] + [Interpolator(plan, d) for d in rife_devs[1:]] if len(rife_devs) > 1 else []
    # 크기 맞추기·필터를 Intel GPU 에서 (4K 에서는 CPU 후처리가 가장 큰 병목이었다)
    gpu_post = gpufilter.make(plan, s, sr_dev)

    # 조각 단위 이어하기 준비 (샘플은 조각 없이 한 번에)
    chunked = duration is None
    job = job_dir_for(path, s, start, duration)
    chunk_len = max(1, int(s.chunk_seconds * plan.fout)) if chunked else total_out
    n_chunks = math.ceil(total_out / chunk_len)
    ext = ".mp4"
    if chunked:
        job.mkdir(parents=True, exist_ok=True)
        done_chunks = {int(p.stem) for p in job.glob("*.done")}
        first = next((c for c in range(n_chunks) if c not in done_chunks), n_chunks)
    else:
        first = 0
    k_start = first * chunk_len
    src_first = int(k_start * plan.ratio)  # 이 출력 프레임에 필요한 첫 원본 프레임

    q_src, q_mid, q_post = queue.Queue(6), queue.Queue(6), queue.Queue(4)
    q_face = queue.Queue(3)  # 업스케일 → 얼굴 복원 (얼굴 복원을 쓸 때만)
    errors = []
    DONE = object()

    def guard(fn):
        def wrapped():
            try:
                fn()
            except Stopped:
                pass
            except Exception as e:  # 다른 스레드도 멈추게 한다
                errors.append(e)
                stop_event.set()
        return wrapped

    def put(q, item):
        while not stop_event.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                pass
        raise Stopped

    def get(q):
        while not stop_event.is_set():
            try:
                return q.get(timeout=0.2)
            except queue.Empty:
                pass
        raise Stopped

    def make_pool(q_in, on_finish):
        """같은 대기열(q_in)에서 일을 가져가는 장치별 작업자 묶음. 빠른 장치가 더 많이 가져가 저절로 균형이 맞는다.
        join(): 작업자 추가 (늦게 준비된 장치는 일이 이미 끝났으면 False) · close(): 더 줄 일이 없음 ·
        run(work): 작업자 본체. 마지막 작업자가 끝나면 on_finish() 로 다음 단계에 끝을 알린다."""
        st = dict(active=0, closed=False)
        lk = threading.Lock()

        def join():
            with lk:
                if st["closed"]:
                    return False
                st["active"] += 1
                return True

        def close():
            with lk:
                st["closed"] = True
            put(q_in, DONE)

        def run(work):
            while True:
                item = get(q_in)
                if item is DONE:
                    put(q_in, DONE)  # 다른 작업자도 끝내도록 되돌려 놓는다
                    break
                work(item)
            with lk:
                st["active"] -= 1
                last = st["active"] == 0
            if last:
                on_finish()
        return join, close, run

    # 보간 작업자 (보간만 모드에서 CPU·GPU 가 나눠 보간)
    q_job = queue.Queue(6)
    interp_join, interp_close, interp_run = make_pool(q_job, lambda: put(q_mid, DONE))

    def interp_worker(ip):
        def work(job):
            k, a, b, t, low = job
            put(q_mid, (k, ip(a, b, t, low=low), None))
        return lambda: interp_run(work)

    def reader():
        if k_start >= total_out:
            put(q_src, DONE)
            return
        r = video_io.FrameReader(info, plan.rw, plan.rh, plan.deint,
                                 start_src + src_first, n_src - src_first)
        try:
            for i, f in enumerate(r):
                put(q_src, (src_first + i, f))
        finally:
            r.close()
        put(q_src, DONE)

    def interpolate():
        k = k_start
        prev, pi = None, None
        dedup = s.dedup and interp is not None
        last_motion, dup_run = 0.0, 0
        # 얼굴 감지는 여기(보간 단계)에서 실제 프레임마다 한 번만 하고, 중간 프레임은 위치를 보간한다.
        # 작은 얼굴이 있는 구간인지 알아야 보간 해상도를 고를 수 있기 때문이기도 하다.
        fdet = face_mod.make_detector() if plan.faces else None
        to_sr = plan.pw * plan.ai_scale / plan.rw  # 원본 좌표 → AI 결과 좌표 배율

        def find_faces(frame):
            return face_mod.detect_frame(fdet, frame) if fdet is not None else None

        def has_small(dets):
            # 복원 대상이면서 원본을 재료로 쓰는 크기(작은 얼굴)가 있는지.
            # 얼굴 수 제한이 있으면 실제로 복원할 큰 얼굴들만 본다
            sizes = sorted((size * to_sr for _, size in dets or [] if size * to_sr >= s.face_min), reverse=True)
            if s.face_max > 0:
                sizes = sizes[:s.face_max]
            return any(sz < face_mod._FULL_STRENGTH_SIZE for sz in sizes)

        def small_centers(dets):
            # 'fast' 모드에서 원본 해상도로 다시 보간할 작은 얼굴들의 중심 (복원할 얼굴만)
            faces = sorted(((size * to_sr, lm) for lm, size in dets or [] if size * to_sr >= s.face_min),
                           key=lambda f: -f[0])
            if s.face_max > 0:
                faces = faces[:s.face_max]
            return [tuple(lm.mean(0)) for sz, lm in faces if sz < face_mod._FULL_STRENGTH_SIZE]

        pdets = cdets = None
        while True:
            item = get(q_src)
            cur, ci = (None, None) if item is DONE else (item[1], item[0])
            if prev is None:
                if cur is None:
                    break
                prev, pi = cur, ci
                pdets = find_faces(prev)
                continue
            if cur is not None and dedup:
                # 중복 프레임은 건너뛰고, 앞뒤 실제 프레임 사이를 시간 간격에 맞춰 보간한다
                d = frame_diff(prev, cur)
                flash = brightness_jump(prev, cur) > 0.03
                if dup_run < 2 and not flash and is_duplicate(d, last_motion):
                    dup_run += 1
                    continue
                dup_run = 0
                if not flash:  # 조명이 크게 바뀐 프레임은 움직임 기준으로 쓰지 않는다
                    last_motion = d / (ci - pi)
            cut = cur is not None and interp is not None and is_scene_cut(prev, cur, s.scene_threshold)
            if cur is not None:
                cdets = find_faces(cur)
            low = s.interp_res in ("proc", "fast") or (s.interp_res == "auto" and not has_small(pdets)
                                                        and not has_small(cdets))
            while k < total_out:
                pos = k * plan.ratio
                if cur is not None and pos >= ci:
                    break
                t = float((pos - pi) / (ci - pi)) if cur is not None else 0.0
                if cur is None or pos < pi or t < 1e-4:
                    f, dets = prev, pdets
                elif cut:
                    f, dets = (prev, pdets) if t < 0.5 else (cur, cdets)
                elif interps:  # 여러 장치가 나눠 보간 (얼굴 복원이 없어 순서가 섞여도 됨)
                    put(q_job, (k, prev, cur, t, low))
                    k += 1
                    continue
                else:
                    f = interp(prev, cur, t, low=low)
                    dets = face_mod.lerp_faces(pdets, cdets, t) if fdet is not None else None
                    if s.interp_res == "fast" and dets:
                        f = interp.refine_faces(f, prev, cur, t, small_centers(dets))
                put(q_mid, (k, f, dets))
                k += 1
            if cur is None:
                break
            prev, pi, pdets = cur, ci, cdets
        if interps:
            interp_close()  # 마지막 보간 작업자가 끝나면 q_mid 에 끝을 알린다
        else:
            put(q_mid, DONE)

    def upscale():
        while True:
            item = get(q_mid)
            if item is DONE:
                break
            k, f, dets = item
            put(q_face if plan.faces else q_post, (k, upscaler.infer(f), f, dets))
        if plan.faces:
            put(q_face, DONE)
        else:
            for _ in range(n_post):
                put(q_post, DONE)

    # 얼굴 복원은 두 단계: 추적(어느 얼굴을 얼마나 복원할지)은 한 스레드가 프레임 순서대로 하고,
    # 실제 복원(GFPGAN)은 장치마다 하나씩인 작업자가 같은 대기열에서 프레임을 가져가 동시에 한다.
    # (Intel GPU 가 꽉 찰 때 노는 NPU 가 일부 프레임을 맡는다. 빠른 장치가 더 많이 가져가 저절로 균형)
    q_restore = queue.Queue(4)

    def finish_post():  # 마지막 복원 작업자가 끝나야 후처리 스레드들을 끝낸다
        for _ in range(n_post):
            put(q_post, DONE)

    restore_join, restore_close, restore_run = make_pool(q_restore, finish_post)

    def faces():
        while True:
            item = get(q_face)
            if item is DONE:
                break
            k, y, f, dets = item
            put(q_restore, (k, y, f, upscaler.plan_faces(y, f, dets)))
        restore_close()

    def restore_worker(run=None):
        def work(item):
            k, y, f, plans = item
            put(q_post, (k, upscaler.restore_faces(y, f, plans, run), f, None))
        return lambda: restore_run(work)

    def extra_restore_worker():
        """보조 장치(NPU) 작업자: 모델 준비(첫 실행 약 1분, 이후 캐시)가 끝나면 합류한다.
        그동안은 기본 장치가 혼자 처리하고, 준비 전에 얼굴 복원이 끝나면 합류하지 않는다."""
        dev = upscaler.face_extra
        try:
            run = _cached_runner(("face-run", dev),
                                 lambda: backends.Runner(face_mod.GFPGAN, dev, (1, 3, face_mod._SIZE, face_mod._SIZE)))
        except Exception:
            log.warning("얼굴 복원 보조 장치 %s 를 쓰지 못함\n%s", dev, traceback.format_exc())
            return
        if not stop_event.is_set() and restore_join():
            guard(restore_worker(run))()

    results = {}
    results_lock = threading.Condition()

    def post():
        while True:
            item = get(q_post)
            if item is DONE:
                break
            k, y, f, _ = item
            out = gpu_post(y, f) if gpu_post else filters.apply_all(*upscaler.finish(y, f), s)
            with results_lock:
                results[k] = out
                results_lock.notify_all()
        with results_lock:
            results[("end", threading.get_ident())] = None
            results_lock.notify_all()

    n_post = 3
    # 처음부터 일하는 작업자는 시작 전에 등록한다 (늦게 합류하는 보조 장치와 구분)
    workers = []
    if plan.faces:
        restore_join()
        workers += [faces, restore_worker()]
    for ip in interps:
        interp_join()
        workers.append(interp_worker(ip))
    threads = [threading.Thread(target=guard(fn), daemon=True)
               for fn in (reader, interpolate, upscale, *workers, *([post] * n_post))]
    for t in threads:
        t.start()
    if plan.faces and upscaler.face_extra:
        # 보조 장치는 준비가 오래 걸릴 수 있어 끝날 때 기다리지 않는다 (threads 에 넣지 않음)
        threading.Thread(target=extra_restore_worker, daemon=True).start()

    # 쓰기는 현재 스레드에서: 순서대로 모아 조각 파일로 인코딩한다
    writer, cur_chunk = None, None
    t0, last_ui = time.time(), 0.0
    done = k_start
    try:
        def open_chunk(c):
            n = min(chunk_len, total_out - c * chunk_len)
            if chunked:
                return video_io.FrameWriter(str(job / f"{c:05d}{ext}"), plan.ow, plan.oh, plan.fout,
                                            encoder, s.crf, s.ten_bit)
            return video_io.FrameWriter(out_path, plan.ow, plan.oh, plan.fout, encoder, s.crf,
                                        s.ten_bit, audio_from=audio_src, audio_codec=info.audio_codec,
                                        audio_start=start, audio_dur=n / float(plan.fout),
                                        audio_mode=s.audio)

        for k in range(k_start, total_out):
            with results_lock:
                while k not in results:
                    if stop_event.is_set():
                        raise Stopped
                    ends = sum(1 for key in results if isinstance(key, tuple))
                    if ends == n_post and k not in results:
                        raise RuntimeError(f"프레임 {k} 가 생성되지 않았습니다.")
                    results_lock.wait(0.2)
                frame = results.pop(k)
                results_lock.notify_all()
            c = k // chunk_len
            if c != cur_chunk:
                if writer:
                    writer.close()
                    if chunked:
                        (job / f"{cur_chunk:05d}.done").touch()
                writer, cur_chunk = open_chunk(c), c
            writer.write(frame)
            done = k + 1
            now = time.time()
            if now - last_ui > 0.5 or done == total_out:
                speed = (done - k_start) / max(1e-6, now - t0)
                eta = (total_out - done) / speed if speed > 0 else 0
                show = frame if now - last_ui > 2 else None
                on_progress(Progress(done, total_out, speed, eta, "처리 중", show))
                last_ui = now
        if writer:
            writer.close()
            if chunked:
                (job / f"{cur_chunk:05d}.done").touch()
            writer = None
    except BaseException as e:
        stop_event.set()
        if writer:
            writer.abort()
        for t in threads:
            t.join(timeout=5)
        if errors:
            raise errors[0]
        if not isinstance(e, Stopped):
            raise
        raise Stopped from None
    for t in threads:
        t.join(timeout=5)
    if errors:
        raise errors[0]

    if chunked:
        on_progress(Progress(total_out, total_out, 0, 0, "영상 합치는 중"))
        parts = [job / f"{c:05d}{ext}" for c in range(n_chunks)]
        video_io.concat(parts, out_path, plan.fout, audio_from=audio_src, audio_codec=info.audio_codec,
                        audio_start=start, list_dir=job, audio_mode=s.audio)
        shutil.rmtree(job, ignore_errors=True)
    on_progress(Progress(total_out, total_out, 0, 0, "완료"))
    return out_path
