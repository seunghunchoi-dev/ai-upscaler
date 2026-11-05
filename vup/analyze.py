"""영상 분석 → 추천 설정.

영상 곳곳에서 프레임을 뽑아 압축 상태·디테일·움직임을 재고, 그에 맞는 설정값과
그렇게 고른 이유를 돌려준다. 화질 취향(스타일)은 건드리지 않고 기술적인 값만 추천한다.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import backends, esrgan, pipeline, video_io
from . import face as face_mod

# 같은 화질을 내는 데 필요한 비트레이트 비율 (H.264 = 1)
_CODEC_EFFICIENCY = {"h264": 1.0, "mpeg4": 0.7, "mpeg2video": 0.5, "vp8": 0.9,
                     "vp9": 1.5, "hevc": 1.6, "av1": 2.0}


@dataclass
class Analysis:
    bpp: float = 0.0            # 실효 픽셀당 비트 (H.264 기준으로 환산)
    damage: float = 0.0         # 압축 손상 점수 (0~1)
    blockiness: float = 1.0     # 8x8 경계 단차 비율 (1 = 블록 없음)
    detail: float = 0.0         # 실제 디테일 (절반 해상도로 줄였다 키웠을 때 사라지는 양)
    noise: float = 0.0          # 노이즈 표준편차 (0~255)
    motion: float = 0.0         # 연속 프레임 평균 차이 (0~1)
    flashes: float = 0.0        # 조명 깜빡임 비율 (0~1)
    duplicates: float = 0.0     # 움직임 중 중복 프레임 비율 (0~1)
    face_frames: float = 0.0    # 얼굴(복원할 만한 크기)이 보이는 샘플 비율 (0~1)
    settings: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


def _gray(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _blockiness(g):
    """8픽셀 경계에서의 밝기 단차가 다른 위치보다 얼마나 큰지."""
    ratios = []
    for d in (np.abs(np.diff(g, axis=1)).mean(axis=0), np.abs(np.diff(g, axis=0)).mean(axis=1)):
        n = len(d) // 8 * 8
        d = d[:n].reshape(-1, 8)
        edge, inner = d[:, 7].mean(), d[:, :7].mean()
        ratios.append(edge / max(inner, 1e-3))
    return float(np.mean(ratios))


def _detail(g):
    h, w = g.shape
    small = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return float(np.abs(g - back).mean())


def _noise(g):
    """Immerkaer 방법으로 추정한 노이즈 표준편차."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    return float(np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1]).sum()
                 * np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)))


def measure(info, samples=8):
    a = Analysis()
    eff = _CODEC_EFFICIENCY.get(info.video_codec, 1.0)
    pixels_per_sec = info.width * info.height * info.fps_float
    a.bpp = info.bitrate * eff / pixels_per_sec if info.bitrate and pixels_per_sec else 0.0

    blocks, details, noises, motions, flashes = [], [], [], [], []
    dups, moving, with_face, sampled = 0, 0, 0, 0
    det = face_mod.make_detector() if face_mod.available() else None
    n = max(1, info.frames)
    for i in range(samples):
        idx = int(n * (i + 0.5) / samples)
        run = list(video_io.FrameReader(info, info.width, info.height, info.interlaced, idx, 6))
        if not run:
            continue
        g0 = _gray(run[0])
        sampled += 1
        if det is not None:
            # 원본 기준 16픽셀 이상이면 업스케일 후 복원할 만한 크기가 된다
            with_face += face_mod.count_faces(det, run[0], 16) > 0
        blocks.append(_blockiness(g0))
        details.append(_detail(g0))
        noises.append(_noise(g0))
        last_motion = 0.0
        for f0, f1 in zip(run, run[1:]):
            if pipeline.is_scene_cut(f0, f1, 0.15):
                last_motion = 0.0
                continue
            d = pipeline.frame_diff(f0, f1)
            jump = pipeline.brightness_jump(f0, f1)
            motions.append(d)
            flashes.append(jump > 0.04)
            if jump > 0.03:  # 조명 깜빡임은 움직임·중복 판단에서 제외 (파이프라인과 같은 규칙)
                continue
            if last_motion > 0.004:
                moving += 1
                if pipeline.is_duplicate(d, last_motion):
                    dups += 1
                    continue  # 중복이면 직전 움직임을 그대로 기준으로 둔다
            last_motion = d
    if blocks:
        a.blockiness = float(np.median(blocks))
        a.detail = float(np.median(details))
        a.noise = float(np.median(noises))
    if motions:
        a.motion = float(np.mean(motions))
        a.flashes = float(np.mean(flashes))
    a.duplicates = dups / moving if moving else 0.0
    a.face_frames = with_face / sampled if sampled else 0.0
    return a


def recommend(info, current=None):
    """영상을 분석해 추천 설정(dict)과 이유를 담은 Analysis 를 돌려준다."""
    a = measure(info)
    s, why = {}, []
    short = min(info.width, info.height)

    # ── 압축 손상 점수 (0~1) → 노이즈 제거 강도 ──
    # 블록 노이즈(8x8 경계 단차)를 주 지표로, 비트레이트는 장면 복잡도에 따라 달라지므로 보조로만 쓴다
    block_score = float(np.clip((a.blockiness - 1.0) / 0.25, 0, 1))
    bpp_score = float(np.clip((0.08 - a.bpp) / 0.07, 0, 1)) if a.bpp else 0.5
    damage = 0.6 * block_score + 0.4 * bpp_score
    if a.noise > 4:
        damage = min(1.0, damage + 0.1)
    a.damage = damage
    s["denoise"] = round(0.35 + 0.6 * damage, 2)
    txt = "심함" if damage > 0.6 else "보통" if damage > 0.3 else "적음"
    br = f"{info.bitrate / 1e6:.1f}Mbps {info.video_codec.upper()}" if info.bitrate else info.video_codec.upper()
    why.append(f"압축 손상: {txt} (블록 노이즈 지수 {a.blockiness:.2f}, {br})"
               f" → 압축 노이즈 제거 {s['denoise'] * 100:.0f}")

    # ── 처리 해상도 ──
    low_quality = damage > 0.6
    if short <= 576:
        s["proc_scale"] = 1.0
        why.append("저해상도 원본 → 처리 해상도 100% (디테일을 모두 사용)")
    elif low_quality:
        s["proc_scale"] = 0.5
        why.append("압축 손상이 커서 → 처리 해상도 50% (줄여서 처리하면 깨짐이 덜 보이고 빠름)")
    else:
        s["proc_scale"] = 0.75
        why.append("압축 손상이 적어서 → 처리 해상도 75% (원본 디테일을 더 살림)")

    # ── 출력 해상도 (유튜브 단계) ──
    if short <= 576:
        target = 1080
    elif short <= 800:
        target = 1440
    elif short <= 1200:
        target = 2160
    else:
        target = 0
    s["target_height"] = target
    if target:
        s["scale"] = 1.0
    # AI 는 처리 해상도의 모델 배율(4배 또는 2배)까지 만들 수 있다 — 넘으면 처리 해상도를 올린다
    model = current.sr_model if current is not None and current.sr_model in esrgan.SR_MODELS else "general_v3"
    need = target / short / esrgan.model_scale(model) if target else 0
    if need > s["proc_scale"]:
        s["proc_scale"] = next((p for p in (0.5, 0.75, 1.0) if p >= need), 1.0)
    why.append(f"원본 {short}p → 출력 {pipeline.resolution_name(target) if target else '원본 크기 유지 (복원만)'}")

    # ── 프레임 ──
    fps = info.fps_float
    if fps < 40:
        s["fps"] = "60"
        why.append(f"{fps:.2f}fps → 60fps 보간 (부드러운 움직임)")
    else:
        s["fps"] = "src"
        why.append(f"{fps:.2f}fps 로 이미 충분히 부드러움 → 보간 안 함")
    s["rife_model"] = "rife_v4.26"

    # ── 장면 전환 / 조명 ──
    if a.flashes > 0.25:
        s["scene_threshold"] = 0.25
        why.append(f"조명 깜빡임 많음 ({a.flashes * 100:.0f}%) → 장면 전환 감지를 둔하게 (조명 변화에 보간이 끊기지 않게)")
    else:
        s["scene_threshold"] = 0.15
    if a.motion > 0.04:
        why.append(f"움직임 많음 (지수 {a.motion:.3f}) → 고품질 보간 모델 RIFE 4.26")
    # ── 얼굴 ──
    if face_mod.available():
        s["face_enhance"] = a.face_frames > 0
        if a.face_frames > 0:
            why.append(f"얼굴이 보이는 장면 {a.face_frames * 100:.0f}% → 얼굴 복원 켬 (체감 화질이 가장 크게 좋아짐)")
        else:
            why.append("복원할 만한 크기의 얼굴이 안 보임 → 얼굴 복원 끔 (속도 우선)")

    s["dedup"] = True
    if a.duplicates > 0.03:
        why.append(f"움직이는 장면에 중복 프레임 {a.duplicates * 100:.0f}% 감지 (방송 녹화·재인코딩 흔적) "
                   "→ 중복 프레임 보정 켬 (줌·이동 중 끊김 방지)")

    # ── 원본이 많이 깨졌으면 원본 질감을 섞지 않는다 ──
    if low_quality and current is not None and current.texture > 0.05:
        s["texture"] = 0.05
        why.append("원본 질감에 블록 노이즈가 섞여 있어 '원본 질감 유지'를 낮춤")

    s["deinterlace"] = "auto"
    if info.interlaced:
        why.append("인터레이스 영상 감지 → 인터레이스 자동 제거")

    # ── 출력 ──
    s["codec"] = "h265"
    s["crf"] = 20 if (target or short) >= 2160 else 18
    s["ten_bit"] = True

    # ── 장치 속도 경고 ──
    sr, _ = pipeline.resolve_devices(pipeline.Settings())
    if sr.startswith(("ov:CPU", "ort:cpu")):
        s["proc_scale"] = 0.5
        why.append("GPU 없음 → 처리 해상도 50% (CPU 처리는 매우 느림)")
    elif "내장" in backends.device_label(sr) and s["proc_scale"] > 0.75:
        why.append("내장 GPU 에서 처리 해상도 100% 는 느릴 수 있음")

    a.settings, a.reasons = s, why
    return a
