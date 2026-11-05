"""AI 처리 뒤에 적용하는 화질 필터. float32 RGB(0~1) 를 받아 같은 형식으로 돌려준다.

무거운 계산은 축소한 영상에서 하고, 원래 크기에서는 `img × M + B` 형태의
가벼운 연산만 하므로 4K 에서도 빠르다. (numpy 대신 멀티코어 OpenCV 연산 사용)
"""
import cv2
import numpy as np

cv2.setUseOptimized(True)


def _up(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _small(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)


def _affine(img, m_small, b_small):
    """img × M + B (M, B 는 축소본에서 계산한 3채널 맵)."""
    h, w = img.shape[:2]
    out = cv2.multiply(img, _up(m_small, w, h))
    return cv2.add(out, _up(b_small, w, h), dst=out)


def keep_texture(sr, base, amount):
    """원본 질감 유지: AI 결과에 원본(단순 확대)을 섞어 인위적인 느낌을 줄인다."""
    if amount <= 0 or base is None:
        return sr
    return cv2.addWeighted(sr, 1.0 - amount, base, amount, 0)


def sharpen(img, amount):
    """언샤프 마스크 기반 선명도."""
    if amount <= 0:
        return img
    sigma = max(0.8, img.shape[0] / 1440)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _skin_mask(small):
    ycc = cv2.cvtColor(np.clip(small, 0, 1), cv2.COLOR_RGB2YCrCb)  # float: Cr,Cb 중심 0.5
    cr, cb = ycc[..., 1] * 255, ycc[..., 2] * 255
    m = (np.clip((cr - 133) / 8, 0, 1) * np.clip((173 - cr) / 8, 0, 1)
         * np.clip((cb - 77) / 8, 0, 1) * np.clip((127 - cb) / 8, 0, 1))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2)


def soft_skin(img, amount, factor=4):
    """피부 보정(뽀샤시): 가이드 필터로 경계는 살리고 면은 매끈하게.
    피부색 영역은 강하게, 나머지는 약하게 적용한다."""
    if amount <= 0:
        return img
    small = _small(img, factor)
    r = max(1, img.shape[0] // 180 // factor)
    k = (2 * r + 1, 2 * r + 1)
    eps = 0.0008 + 0.004 * amount
    mean = cv2.boxFilter(small, -1, k)
    var = cv2.boxFilter(small * small, -1, k) - mean * mean
    a = var / (var + eps)
    b = mean * (1.0 - a)
    a, b = cv2.boxFilter(a, -1, k), cv2.boxFilter(b, -1, k)
    wgt = ((_skin_mask(small) * 0.85 + 0.15) * min(1.0, amount))[..., None]
    # 결과 = img + w·(a·img + b − img) = img·(1 + w(a−1)) + w·b
    return _affine(img, 1.0 + wgt * (a - 1.0), wgt * b)


def glow(img, amount):
    """빛 번짐(블룸): 밝은 부분이 은은하게 퍼진다. screen 합성 = img·(1−g) + g."""
    if amount <= 0:
        return img
    small = _small(img, 8)
    bright = np.clip((small - 0.55) / 0.45, 0, 1)
    g = cv2.GaussianBlur(bright, (0, 0), max(2.0, small.shape[0] / 60)) * (0.6 * amount)
    return _affine(img, 1.0 - g, g)


_noise_pool = {}


def grain_noise(w, h):
    """이번 프레임의 그레인 노이즈 (h//2, w//2) 1채널. GPU 후처리(gpufilter)도 같은 노이즈를 쓴다."""
    key = (w, h)
    if key not in _noise_pool:
        rng = np.random.default_rng(1)
        pool = [rng.standard_normal((h // 2, w // 2), dtype=np.float32) for _ in range(6)]
        _noise_pool.clear()
        _noise_pool[key] = [pool, 0]
    pool, i = _noise_pool[key]
    _noise_pool[key][1] = (i + 1) % len(pool)
    # 매 프레임 다른 위치로 뒤집어 반복 무늬가 보이지 않게
    return cv2.flip(pool[i], (i % 3) - 1) if i % 2 else pool[i]


def grain(img, amount):
    """필름 그레인: 미세한 입자로 자연스러운 질감을 더한다."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    n = grain_noise(w, h)
    return cv2.scaleAdd(_up(cv2.merge([n, n, n]), w, h), 0.025 * amount, img)


def clarity(img, amount):
    """명료도(입체감): 넓은 범위의 밝기 대비를 살짝 키운다 (사진 앱의 '명료도'와 같은 효과).
    축소본에서 넓게 흐린 밝기를 빼고 더하므로 4K 에서도 가볍다."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    small = _small(img, 8)
    blur = _up(cv2.GaussianBlur(small, (0, 0), max(2.0, small.shape[0] / 40)), w, h)
    return cv2.addWeighted(img, 1.0 + 0.6 * amount, blur, -0.6 * amount, 0)


def vibrance(img, amount):
    """생생함: 채도가 낮은 색 위주로 채도를 올린다 (이미 진한 색·피부는 덜 과해짐)."""
    if amount <= 0:
        return img
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    sat = cv2.max(cv2.max(img[..., 0], img[..., 1]), img[..., 2]) - cv2.min(cv2.min(img[..., 0], img[..., 1]), img[..., 2])
    gain = (1.0 + 0.8 * amount * (1.0 - np.clip(sat, 0, 1)))[..., None]
    return gray + (img - gray) * gain


def _brighten_lut(amount):
    x = np.arange(256, dtype=np.float32) / 255
    return (np.power(x, 1.0 - 0.25 * amount) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def brighten_u8(img, amount):
    """화사함: 중간 밝기를 살짝 올리고(LUT) 채도를 조금 높인다. uint8 입력."""
    if amount <= 0:
        return img
    out = cv2.LUT(img, _brighten_lut(amount))
    gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    k = 0.2 * amount
    return cv2.addWeighted(out, 1.0 + k, gray, -k, 0)


def to_u8(img):
    return cv2.convertScaleAbs(cv2.max(img, 0.0), alpha=255.0)


def apply_all(sr, base, s):
    """설정(s)에 따라 필터를 순서대로 적용하고 uint8 로 돌려준다."""
    img = keep_texture(sr, base, s.texture)
    img = soft_skin(img, s.soft_skin)
    img = sharpen(img, s.sharpen)
    img = clarity(img, getattr(s, "clarity", 0.0))
    img = vibrance(img, getattr(s, "vibrance", 0.0))
    img = glow(img, s.glow)
    img = grain(img, s.grain)
    return brighten_u8(to_u8(img), s.brighten)
