"""얼굴 복원 (GFPGAN 1.4) — 음악방송처럼 얼굴이 중요한 영상의 체감 화질을 크게 올린다.

  1) YuNet 으로 얼굴과 눈·코·입 위치(5점)를 찾고
  2) 표준 얼굴 위치에 맞춰 512x512 로 잘라 GFPGAN 으로 복원한 뒤
  3) 부드러운 가장자리 마스크로 원래 자리에 되돌려 붙인다.

영상에서는 프레임마다 따로 복원하면 얼굴이 깜빡이므로, 얼굴을 추적해 위치를 부드럽게 잇고
새로 나타나거나 사라지는 얼굴은 복원 강도를 서서히 바꾼다.
"""
import cv2
import numpy as np

from . import backends
from .paths import MODELS

try:  # OpenCV dnn 의 의미 없는 경고("Targets are not supported...") 숨김
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

GFPGAN = MODELS / "gfpgan_1.4.onnx"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"

# GFPGAN(FFHQ) 512 기준 표준 위치: 화면 왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입꼬리, 오른쪽 입꼬리
_TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                      [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
_SIZE = 512
# 감지할 때 줄이는 가로 크기. 960 에서는 4K 화면의 64px 미만 얼굴(원본 1080p 기준 32px 미만)을
# 대부분 놓쳤고, 1920 에서는 32px(원본 16px) 얼굴까지 모두 찾았다.
_DET_WIDTH = 1920
# 이 크기(px) 이상인 얼굴은 설정한 강도 그대로, 작을수록 약하게 복원한다
_FULL_STRENGTH_SIZE = 128
_MIN_STRENGTH_RATIO = 0.45


def size_ratio(size, min_size):
    """얼굴 크기를 0(최소 크기) ~ 1(충분히 큼) 로."""
    span = max(1.0, _FULL_STRENGTH_SIZE - min_size)
    return float(np.clip((size - min_size) / span, 0.0, 1.0))


def size_factor(size, min_size):
    """작은 얼굴은 정보가 적어 세게 복원하면 다른 얼굴처럼 보이거나 깜빡이므로 강도를 낮춘다."""
    return _MIN_STRENGTH_RATIO + (1.0 - _MIN_STRENGTH_RATIO) * size_ratio(size, min_size)


def available():
    return GFPGAN.exists() and YUNET.exists()


_detector_buf = None


def make_detector():
    """YuNet 얼굴 감지기. 한글 경로에서도 읽히도록 파일 내용을 메모리로 넘긴다."""
    global _detector_buf
    if _detector_buf is None:
        _detector_buf = np.fromfile(str(YUNET), np.uint8)
    return cv2.FaceDetectorYN.create("onnx", _detector_buf, np.array([], np.uint8), (320, 320), 0.6, 0.3, 50)


def detect_frame(det, frame):
    """frame(uint8 또는 float RGB)의 얼굴들을 [(눈코입 5점, 크기)] 로 (frame 좌표 기준)."""
    h, w = frame.shape[:2]
    s = min(1.0, _DET_WIDTH / w)
    small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
    if small.dtype != np.uint8:
        small = np.clip(small * 255, 0, 255).astype(np.uint8)
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None:
        return []
    return [(f[4:14].reshape(5, 2) / s, float(max(f[2], f[3]) / s)) for f in faces]


def lerp_faces(a, b, t):
    """중간 프레임의 얼굴 위치: 앞(a)·뒤(b) 프레임에서 같은 얼굴끼리 이어 t 비율로 보간한다.
    (중간 프레임마다 다시 감지하지 않아 CPU 를 아낀다)"""
    out, used = [], set()
    for lm, size in a:
        c = lm.mean(0)
        best, bd = None, 0.5 * size
        for j, (lm2, size2) in enumerate(b):
            d = float(np.linalg.norm(lm2.mean(0) - c))
            if j not in used and d < bd:
                best, bd = j, d
        if best is None:
            if t < 0.5:
                out.append((lm, size))
        else:
            used.add(best)
            lm2, size2 = b[best]
            out.append((lm + (lm2 - lm) * t, size + (size2 - size) * t))
    if t >= 0.5:
        out += [f for j, f in enumerate(b) if j not in used]
    return out


def count_faces(det, frame_rgb_u8, min_size):
    """영상 분석용: 프레임에서 min_size 픽셀 이상인 얼굴 수."""
    h, w = frame_rgb_u8.shape[:2]
    s = min(1.0, _DET_WIDTH / w)
    small = cv2.resize(frame_rgb_u8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    return 0 if faces is None else int(sum(max(f[2], f[3]) / s >= min_size for f in faces))


def _feather_mask():
    m = np.zeros((_SIZE, _SIZE), np.float32)
    cv2.rectangle(m, (40, 40), (_SIZE - 40, _SIZE - 40), 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 18)


class FaceEnhancer:
    def __init__(self, device, strength=0.7, min_size=32):
        self.strength, self.min_size = float(strength), int(min_size)
        self.max_faces = 0  # 한 화면에서 복원할 최대 얼굴 수 (0 = 제한 없음, 큰 얼굴부터)
        self.run = backends.Runner(GFPGAN, device, (1, 3, _SIZE, _SIZE))
        self.det = make_detector()
        self.mask = _feather_mask()
        self.tracks = []  # dict(lm=5x2, size, age, miss)

    def reset(self):
        self.tracks = []

    def _limit(self, faces, key):
        """복원할 얼굴 수 제한: 큰 얼굴부터 max_faces 개 (단체 샷에서 GPU 시간을 줄임)."""
        if self.max_faces <= 0 or len(faces) <= self.max_faces:
            return faces
        return sorted(faces, key=key, reverse=True)[:self.max_faces]

    def _scale(self, dets, src_w, scale_to):
        """frame 좌표의 얼굴들을 결과 이미지(scale_to 너비) 좌표로 바꾸고 너무 작은 얼굴은 뺀다."""
        k = scale_to / src_w
        return [(lm * k, size * k) for lm, size in dets if size * k >= self.min_size]

    def _detect(self, frame, scale_to):
        """frame(uint8 또는 float RGB)에서 얼굴을 찾아 scale_to 너비 기준 좌표로 돌려준다."""
        return self._scale(detect_frame(self.det, frame), frame.shape[1], scale_to)

    def _update_tracks(self, dets):
        for t in self.tracks:
            t["matched"] = False
        for lm, size in dets:
            c = lm.mean(0)
            best, bd = None, 0.5 * size
            for t in self.tracks:
                d = float(np.linalg.norm(t["lm"].mean(0) - c))
                if not t["matched"] and d < bd:
                    best, bd = t, d
            if best is None:
                self.tracks.append(dict(lm=lm, size=size, age=1, miss=0, matched=True))
            else:
                # 작은 흔들림은 부드럽게, 큰 움직임은 바로 따라간다
                a = 0.5 if bd < 0.05 * size else 0.85
                best.update(lm=best["lm"] * (1 - a) + lm * a, size=best["size"] * (1 - a) + size * a,
                            age=best["age"] + 1,
                            miss=0, matched=True)
        for t in self.tracks:
            if not t["matched"]:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= 2]

    def _restore(self, img, lm, strength, ratio=1.0, src=None, run=None):
        """ratio: 얼굴 크기 비율(0 작음 ~ 1 큼). src: AI 처리 전 원본 프레임(uint8, 더 작은 해상도).
        run: GFPGAN 을 돌릴 Runner (없으면 기본 장치). 영상에서는 GPU 와 NPU 가 프레임을 나눠 복원한다.

        작은 얼굴은 AI 업스케일이 형태를 망가뜨리는 경우가 많아(녹아내린 듯한 얼굴),
        흐릿하지만 형태가 온전한 원본 얼굴을 복원 재료로 쓴다. 큰 얼굴은 더 선명한 AI 결과를 쓴다."""
        h, w = img.shape[:2]
        M, _ = cv2.estimateAffinePartial2D(lm, _TEMPLATE, method=cv2.LMEDS)
        if M is None:
            return
        crop = cv2.warpAffine(img, M, (_SIZE, _SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if src is not None and ratio < 1.0:
            # 원본 프레임에서 같은 얼굴을 바로 512 로 잘라 온다 (전체를 키우지 않음)
            Ms, _ = cv2.estimateAffinePartial2D(lm * (src.shape[1] / w), _TEMPLATE, method=cv2.LMEDS)
            if Ms is not None:
                orig = cv2.warpAffine(src, Ms, (_SIZE, _SIZE), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REFLECT101).astype(np.float32) * (1 / 255)
                crop = orig * (1.0 - ratio) + crop * ratio
        x = (crop.transpose(2, 0, 1)[None] - 0.5) / 0.5
        y = (run or self.run)(np.ascontiguousarray(x, np.float32))[0]
        restored = (np.clip(y, -1, 1).transpose(1, 2, 0) + 1) / 2
        restored = crop + (restored - crop) * strength

        # 얼굴이 들어갈 영역만 계산해서 되돌려 붙인다 (4K 전체를 변환하지 않음)
        Minv = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0, 1], [_SIZE, 0, 1], [0, _SIZE, 1], [_SIZE, _SIZE, 1]], np.float32) @ Minv.T
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        Mroi = Minv.copy()
        Mroi[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        face = cv2.warpAffine(restored, Mroi, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(self.mask, Mroi, size, flags=cv2.INTER_LINEAR)[..., None]
        roi = img[y0:y1, x0:x1]
        roi += (face - roi) * mask

    def plan(self, img, src=None, dets=None, track=True):
        """복원할 얼굴들 [(5점, 강도, 크기 비율)] (img 좌표). 추적을 하므로 영상에서는 프레임 순서대로 부른다.
        src: AI 처리 전 원본 프레임(uint8 RGB). 얼굴 감지와 작은 얼굴의 복원 재료로 쓴다.
        (AI 업스케일은 작은 얼굴을 망가뜨려 감지조차 안 되는 경우가 있어서 원본에서 찾는다)
        dets: 미리 찾아 둔 얼굴 [(5점, 크기)] (src 좌표). 없으면 여기서 찾는다."""
        if self.strength <= 0:
            return []
        base = src if src is not None else img
        if dets is None:
            dets = self._detect(base, img.shape[1])
        else:
            dets = self._scale(dets, base.shape[1], img.shape[1])
        if not track:
            return [(lm, self.strength * size_factor(size, self.min_size), size_ratio(size, self.min_size))
                    for lm, size in self._limit(dets, key=lambda f: f[1])]
        self._update_tracks(dets)
        out = []
        for t in self._limit(self.tracks, key=lambda t: t["size"]):
            fade = min(1.0, t["age"] / 4) * (1.0 - t["miss"] / 3)  # 서서히 나타나고 서서히 사라짐
            if fade > 0.05:
                out.append((t["lm"].copy(), self.strength * fade * size_factor(t["size"], self.min_size),
                            size_ratio(t["size"], self.min_size)))
        return out

    def apply(self, img, plans, src=None, run=None):
        """plan() 결과대로 얼굴을 복원한다. 프레임끼리는 순서와 상관없이 따로(동시에) 처리할 수 있다."""
        img = np.ascontiguousarray(img)
        for lm, strength, ratio in plans:
            self._restore(img, lm, strength, ratio, src, run)
        return img

    def __call__(self, img, track=True, src=None, dets=None):
        """img: float32 RGB (0~1), 제자리에서 수정해 돌려준다. track=False 면 한 장짜리(미리보기)."""
        img = np.ascontiguousarray(img)
        return self.apply(img, self.plan(img, src, dets, track), src)
