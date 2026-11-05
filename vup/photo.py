"""사진 개선: 설정, 자동 분석(추천값), 처리, 저장."""
import struct
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from . import analyze, backends, esrgan, filters
from . import face as face_mod

TILE = 512          # AI 에 넣는 조각 크기 (고정 크기라 모델을 한 번만 준비하면 됨)
MARGIN = 16         # 조각 경계 이음매를 없애기 위해 겹쳐 처리하는 여백
MAX_PIXELS = 70_000_000  # 결과 사진 최대 화소 (이보다 크면 메모리 부족 위험)


@dataclass
class PhotoSettings:
    quality: str = "general_v3"   # general_v3(표준) | x4plus(고품질)
    size: object = "x2"           # "x1" | "x2" | "x4" | 긴 변 픽셀(int) | "custom"
    size_custom: int = 3840       # size == "custom" 일 때 긴 변
    denoise: float = 0.5
    face_enhance: bool = True
    face_strength: float = 0.7
    # 스타일 (기본 = '깨끗하게')
    texture: float = 0.05
    soft_skin: float = 0.2
    glow: float = 0.0
    brighten: float = 0.0
    sharpen: float = 0.2
    grain: float = 0.0
    # 저장
    fmt: str = "same"             # same | .jpg | .png | .webp
    jpg_quality: int = 95
    keep_exif: bool = True

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self):
        return asdict(self)

    def output_size(self, w, h):
        if self.size in ("x1", "x2", "x4"):
            k = int(self.size[1])
            return w * k, h * k
        long_side = self.size_custom if self.size == "custom" else int(self.size)
        k = long_side / max(w, h)
        return int(round(w * k)), int(round(h * k))


    def size_label(self):
        if self.size == "x1":
            return "원본크기"
        if self.size in ("x2", "x4"):
            return self.size
        return f"{self.size_custom if self.size == 'custom' else self.size}px"


# 자동 설정이 바꾸는 항목 (스타일 취향과 저장 형식은 건드리지 않음)
AUTO_KEYS = ("quality", "size", "size_custom", "denoise", "face_enhance", "face_strength", "texture")

_detector = None


def recommend(img, current=None):
    """사진(RGB numpy)을 분석해 (추천 설정 dict, 이유 목록)."""
    global _detector
    h, w = img.shape[:2]
    long_side, pixels = max(w, h), w * h
    s, why = {}, []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 압축 깨짐 → 노이즈 제거
    block = analyze._blockiness(gray)
    noise = analyze._noise(gray)
    damage = float(np.clip((block - 1.0) / 0.25, 0, 1))
    if noise > 4:
        damage = min(1.0, damage + 0.15)
    s["denoise"] = round(0.3 + 0.6 * damage, 2)
    level = "심함" if damage > 0.6 else "보통" if damage > 0.3 else "적음"
    why.append(f"압축 깨짐 {level} (블록 지수 {block:.2f}, 노이즈 {noise:.1f}) → 노이즈 제거 {s['denoise'] * 100:.0f}")
    s["texture"] = 0.0 if damage > 0.6 else 0.05

    # 크기 → 출력 크기 (결과가 8K 를 넘지 않게)
    if long_side >= 3000:
        s["size"] = "x1"
        why.append(f"이미 큰 사진 ({w}x{h}) → 크기는 그대로, 화질만 복원")
    elif long_side >= 1500:
        s["size"] = "x2"
        why.append(f"중간 크기 ({w}x{h}) → 2배")
    elif long_side * 4 <= 7680:
        s["size"] = "x4"
        why.append(f"작은 사진 ({w}x{h}) → 4배")
    else:
        s["size"] = 7680
        why.append(f"작은 사진 ({w}x{h}) → 긴 변 8K(7680px)까지")

    # 사진 크기 → 품질 모델 (고품질은 픽셀 수에 비례해 느려짐)
    mp = pixels / 1e6
    if pixels <= 2_100_000 and "x4plus" in esrgan.available_models():
        s["quality"] = "x4plus"
        why.append(f"{mp:.1f}메가픽셀 → 고품질 모델 (디테일 복원력이 좋고 이 크기면 빠름)")
    else:
        s["quality"] = "general_v3"
        why.append(f"{mp:.1f}메가픽셀 → 표준 모델 (이 크기에서 고품질 모델은 너무 느림)")

    # 얼굴 → 얼굴 복원
    if face_mod.available():
        if _detector is None:
            _detector = face_mod.make_detector()
        faces = [f for f in face_mod.detect_frame(_detector, img) if f[1] >= 16]
        s["face_enhance"] = bool(faces)
        if faces:
            biggest = max(f[1] for f in faces)
            s["face_strength"] = 0.7 if biggest >= 64 else 0.5
            why.append(f"얼굴 {len(faces)}개 (가장 큰 얼굴 {biggest:.0f}px) → 얼굴 복원 켬, 강도 "
                       f"{s['face_strength'] * 100:.0f}" + ("" if biggest >= 64 else " (작은 얼굴이라 약하게)"))
        else:
            why.append("얼굴 없음 → 얼굴 복원 끔")
    return s, why


# ── 처리 ─────────────────────────────────────────────────────────────────

class Stopped(Exception):
    pass


_runners = {}


def _runner(key, make):
    if key not in _runners:
        if len(_runners) >= 3:
            _runners.pop(next(iter(_runners)))
        _runners[key] = make()
    return _runners[key]


def enhance(img, s, device="auto", on_progress=None, stop_event=None):
    """사진(RGB uint8)을 설정 s 대로 개선해 RGB uint8 로 돌려준다.

    512 조각으로 나눠 AI 업스케일하고, 각 조각을 바로 목표 크기로 줄여 붙인다
    (원본 크기로 저장할 때도 4배 크기의 거대한 중간 이미지를 만들지 않음).
    on_progress(0~100, 설명) 으로 진행률을 알린다."""
    on_progress = on_progress or (lambda p, t: None)
    if device == "auto":
        device = backends.auto_device()
    h, w = img.shape[:2]
    ow, oh = s.output_size(w, h)
    if ow * oh > MAX_PIXELS:
        raise RuntimeError(f"결과 크기 {ow}x{oh} ({ow * oh / 1e6:.0f}메가픽셀) 가 너무 큽니다. "
                           f"출력 크기를 줄여 주세요 (최대 약 {MAX_PIXELS / 1e6:.0f}메가픽셀).")
    model = s.quality if s.quality in esrgan.available_models() else "general_v3"
    scale = esrgan.model_scale(model)
    dn = round(s.denoise, 2) if model == "general_v3" else None
    run = _runner(("sr", model, dn, device),
                  lambda: backends.Runner(esrgan.build_onnx(s.denoise, model), device, (1, TILE, TILE, 3)))

    canvas = np.zeros((oh, ow, 3), np.float32)
    sx, sy = ow / w, oh / h
    core = TILE - 2 * MARGIN
    ys, xs = list(range(0, h, core)), list(range(0, w, core))
    total, done = len(ys) * len(xs), 0
    for y0 in ys:
        for x0 in xs:
            if stop_event is not None and stop_event.is_set():
                raise Stopped
            y1, x1 = min(y0 + core, h), min(x0 + core, w)
            iy0, ix0 = max(0, y0 - MARGIN), max(0, x0 - MARGIN)
            iy1, ix1 = min(h, y1 + MARGIN), min(w, x1 + MARGIN)
            tile = img[iy0:iy1, ix0:ix1]
            tile = cv2.copyMakeBorder(tile, 0, TILE - tile.shape[0], 0, TILE - tile.shape[1], cv2.BORDER_REPLICATE)
            out = run(np.ascontiguousarray(tile)[None])[0]
            core_out = out[(y0 - iy0) * scale:(y1 - iy0) * scale, (x0 - ix0) * scale:(x1 - ix0) * scale]
            ty0, ty1 = round(y0 * sy), round(y1 * sy)
            tx0, tx1 = round(x0 * sx), round(x1 * sx)
            if ty1 > ty0 and tx1 > tx0:
                shrink = (tx1 - tx0) < core_out.shape[1]
                canvas[ty0:ty1, tx0:tx1] = cv2.resize(core_out, (tx1 - tx0, ty1 - ty0),
                                                      interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
            done += 1
            on_progress(5 + 75 * done / total, f"AI 복원 {done}/{total}")

    if s.face_enhance and face_mod.available():
        on_progress(82, "얼굴 복원")
        fe = _runner(("face", device), lambda: face_mod.FaceEnhancer(device))
        fe.strength, fe.min_size = s.face_strength, 32
        fe(canvas, track=False, src=img)
    on_progress(92, "마무리")
    base = None
    if s.texture > 0:
        base = cv2.multiply(cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC), 1 / 255, dtype=cv2.CV_32F)
    result = filters.apply_all(canvas, base, s)
    on_progress(100, "완료")
    return result


# ── 저장 ─────────────────────────────────────────────────────────────────

def output_path(src, s, out_dir=""):
    p = Path(src)
    ext = p.suffix.lower() if s.fmt == "same" else s.fmt
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".png"  # BMP·TIFF 원본은 무손실 PNG 로
    folder = Path(out_dir) if out_dir.strip() else p.parent
    base = folder / f"{p.stem}_개선_{s.size_label()}"
    out, n = Path(str(base) + ext), 1
    while out.exists():
        out = Path(f"{base} ({n}){ext}")
        n += 1
    return out


def _jpeg_exif(data):
    """JPEG 바이트에서 EXIF(APP1) 세그먼트를 찾아 돌려준다. 없으면 None."""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
            return bytearray(data[i:i + 2 + length])
        if marker == 0xDA:  # 영상 데이터 시작 — 더 찾지 않음
            break
        i += 2 + length
    return None


def _reset_orientation(seg):
    """EXIF 의 회전값(Orientation)을 1(정상)로 바꾼다. 회전은 이미 픽셀에 적용했기 때문."""
    t = 10  # APP1 헤더(4) + "Exif\0\0"(6) 다음이 TIFF 시작
    if len(seg) < t + 8:
        return seg
    fmt = "<" if seg[t:t + 2] == b"II" else ">"
    ifd = t + struct.unpack(fmt + "I", seg[t + 4:t + 8])[0]
    n = struct.unpack(fmt + "H", seg[ifd:ifd + 2])[0]
    for k in range(n):
        e = ifd + 2 + 12 * k
        if struct.unpack(fmt + "H", seg[e:e + 2])[0] == 0x0112:
            seg[e + 8:e + 10] = struct.pack(fmt + "H", 1)
    return seg


def save(result, src, s, out_dir=""):
    """결과를 저장하고 경로를 돌려준다. JPG→JPG 는 촬영 정보(EXIF)를 옮긴다."""
    out = output_path(src, s, out_dir)
    ext = out.suffix.lower()
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, s.jpg_quality])
    elif ext == ".webp":
        ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, s.jpg_quality])
    else:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("사진 저장(인코딩) 실패")
    data = buf.tobytes()
    if s.keep_exif and ext in (".jpg", ".jpeg"):
        try:
            exif = _jpeg_exif(Path(src).read_bytes())
            if exif:
                data = data[:2] + bytes(_reset_orientation(exif)) + data[2:]
        except Exception:
            pass  # 촬영 정보가 깨져 있어도 사진 저장은 계속
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out
