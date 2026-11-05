"""후처리(크기 맞추기 + filters.apply_all)를 Intel GPU 에서 실행한다.

4K 출력에서는 AI 업스케일보다 CPU 후처리가 더 오래 걸렸다 (실측: 프레임당 약 1.2초, GPU 는 70% 놀고 있음).
filters.py 와 같은 계산(같은 커널·순서·경계 처리)을 ONNX 그래프로 만들어 OpenVINO GPU 에서 돌리고,
결과를 uint8 로 받아 CPU 와 메모리 대역폭을 아낀다. 쓸 수 없는 조건이면 make() 가 None 을 돌려주고
기존 CPU 경로를 쓴다.
"""
import threading

import cv2
import numpy as np
from onnx import TensorProto, helper

from . import backends, filters
from .esrgan import _Graph


def _cubic_taps(n_in, n_out):
    """cv2.resize(INTER_CUBIC) 의 한 축: 출력 위치마다 원본 4픽셀의 번호(경계는 끝 픽셀)와 가중치."""
    A = -0.75
    fx = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
    sx = np.floor(fx)
    x = (fx - sx).astype(np.float32)
    c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
    c1 = ((A + 2) * x - (A + 3)) * x * x + 1
    c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1
    c3 = 1 - c0 - c1 - c2
    idx = np.clip(sx[:, None].astype(np.int64) + np.arange(-1, 3), 0, n_in - 1)
    return idx, np.stack([c0, c1, c2, c3], 1).astype(np.float32)


def _area_taps(n_in, n_out):
    """cv2.resize(INTER_AREA) 축소의 한 축 (OpenCV computeResizeAreaTab 과 같은 면적 가중치)."""
    scale = n_in / n_out
    rows = []
    for d in range(n_out):
        f1 = d * scale
        f2 = f1 + scale
        s1, s2 = int(np.ceil(f1)), int(np.floor(f2))
        s2 = min(s2, n_in - 1)
        cell = min(scale, n_in - f1)
        taps = []
        if s1 - f1 > 1e-3:
            taps.append((s1 - 1, (s1 - f1) / cell))
        taps += [(s, 1 / cell) for s in range(s1, s2)]
        if f2 - s2 > 1e-3:
            taps.append((s2, min(min(f2 - s2, 1.0), cell) / cell))
        rows.append(taps)
    n = max(len(t) for t in rows)
    idx = np.zeros((n_out, n), np.int64)
    wts = np.zeros((n_out, n), np.float32)
    for d, taps in enumerate(rows):
        for j, (s, wt) in enumerate(taps):
            idx[d, j], wts[d, j] = s, wt
    return idx, wts


def _gauss_kernel(sigma):
    """cv2.GaussianBlur(float 이미지, (0,0), sigma) 와 같은 1차원 커널."""
    k = int(round(sigma * 8 + 1)) | 1
    return cv2.getGaussianKernel(k, sigma, cv2.CV_32F).ravel()


class _G(_Graph):
    """NCHW float 이미지 연산 (경계는 cv2 기본값인 BORDER_REFLECT_101 = ONNX reflect)."""

    def pad(self, x, ph, pw):
        return self.op("Pad", [x, self.const(np.array([0, 0, ph, pw, 0, 0, ph, pw], np.int64))], mode="reflect")

    def sep_conv(self, x, k1d, ch):
        n = len(k1d)
        x = self.pad(x, n // 2, n // 2)
        kv = np.tile(k1d.reshape(1, 1, n, 1), (ch, 1, 1, 1)).astype(np.float32)
        kh = np.tile(k1d.reshape(1, 1, 1, n), (ch, 1, 1, 1)).astype(np.float32)
        x = self.op("Conv", [x, self.const(kv)], group=ch, kernel_shape=[n, 1])
        return self.op("Conv", [x, self.const(kh)], group=ch, kernel_shape=[1, n])

    def gauss(self, x, sigma, ch=3):
        return self.sep_conv(x, _gauss_kernel(sigma), ch)

    def box(self, x, k, ch=3):
        return self.sep_conv(x, np.full(k, 1.0 / k, np.float32), ch)

    def area_down(self, x, f):
        return self.op("AveragePool", [x], kernel_shape=[f, f], strides=[f, f])

    def resize(self, x, h, w, mode):
        # 채널 수는 입력과 같게: sizes 의 1번 값을 입력에서 가져온다
        ch = self.op("Slice", [self.op("Shape", [x]), self.const(np.array([1], np.int64)),
                               self.const(np.array([2], np.int64))])
        sizes = self.op("Concat", [self.const(np.array([1], np.int64)), ch,
                                   self.const(np.array([h, w], np.int64))], axis=0)
        attrs = dict(mode=mode, coordinate_transformation_mode="half_pixel")
        if mode == "cubic":
            attrs["cubic_coeff_a"] = -0.75  # OpenCV INTER_CUBIC 와 같은 계수
        return self.op("Resize", [x, "", "", sizes], **attrs)

    def taps_resize(self, x, src_hw, h, w, taps):
        """축마다 원본 픽셀 몇 개를 모아(Gather) 가중합하는 크기 조정.
        ONNX Resize(cubic) 는 Intel GPU 에서 참조 구현으로 돌아 4K 한 장에 3초가 걸렸다."""
        for axis, n_in, n_out in ((3, src_hw[1], w), (2, src_hw[0], h)):
            if n_in == n_out:
                continue
            idx, wts = taps(n_in, n_out)
            shape = [1, 1, 1, 1]
            shape[axis] = n_out
            acc = None
            for k in range(idx.shape[1]):
                t = self.mul(self.op("Gather", [x, self.const(idx[:, k])], axis=axis),
                             self.const(wts[:, k].reshape(shape)))
                acc = t if acc is None else self.add(acc, t)
            x = acc
        return x

    def cubic(self, x, src_hw, h, w):
        """cv2 INTER_CUBIC 확대."""
        return self.taps_resize(x, src_hw, h, w, _cubic_taps)

    def area(self, x, src_hw, h, w):
        """cv2 INTER_AREA 축소 (정수배가 아니어도 같은 결과)."""
        return self.taps_resize(x, src_hw, h, w, _area_taps)

    def c(self, v):
        return self.const(np.array(v, np.float32))

    def mul(self, a, b):
        return self.op("Mul", [a, b])

    def add(self, a, b):
        return self.op("Add", [a, b])

    def sub(self, a, b):
        return self.op("Sub", [a, b])

    def clip01(self, x, lo=0.0, hi=1.0):
        return self.op("Clip", [x, self.c(lo), self.c(hi)])

    def weighted(self, a, wa, b, wb):
        """cv2.addWeighted(a, wa, b, wb, 0)."""
        return self.add(self.mul(a, self.c(wa)), self.mul(b, self.c(wb)))

    def channel(self, x, i):
        return self.op("Slice", [x, self.const(np.array([i], np.int64)), self.const(np.array([i + 1], np.int64)),
                                 self.const(np.array([1], np.int64))])

    def gray(self, x):
        """cv2 RGB2GRAY (float) → 1채널."""
        w = self.const(np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1))
        return self.op("ReduceSum", [self.mul(x, w), self.const(np.array([1], np.int64))], keepdims=1)

    def affine(self, img, m_small, b_small, h, w):
        """filters._affine: img × up(M) + up(B)."""
        return self.add(self.mul(img, self.resize(m_small, h, w, "linear")), self.resize(b_small, h, w, "linear"))


def _build(s, y_shape, frame_shape, oh, ow, need_base):
    """y_shape: AI 결과 (H, W) 또는 None, frame_shape: 읽기 해상도 (h, w)."""
    g = _G()
    inputs = []
    base = None
    if need_base:
        inputs.append(helper.make_tensor_value_info("frame", TensorProto.UINT8, [1, *frame_shape, 3]))
        f = g.op("Transpose", [g.op("Cast", ["frame"], to=TensorProto.FLOAT)], perm=[0, 3, 1, 2])
        if tuple(frame_shape) != (oh, ow):
            # cv2.resize(uint8, INTER_CUBIC) 는 결과를 0~255 정수로 저장한다
            f = g.op("Round", [g.clip01(g.cubic(f, frame_shape, oh, ow), 0.0, 255.0)])
        base = g.mul(f, g.c(1 / 255))
    if y_shape is not None:
        inputs.append(helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, *y_shape, 3]))
        img = g.op("Transpose", ["y"], perm=[0, 3, 1, 2])
        H, W = y_shape
        if (H, W) != (oh, ow):
            if W > ow:  # Upscaler.finish 와 같은 기준: 가로가 출력보다 크면 INTER_AREA
                img = g.area(img, (H, W), oh, ow)
            else:
                img = g.cubic(img, (H, W), oh, ow)
        if s.texture > 0:
            img = g.weighted(img, 1.0 - s.texture, base, s.texture)
    else:
        img = base

    h, w = oh, ow
    if s.soft_skin > 0:
        fct = 4
        small = g.area_down(img, fct)
        r = max(1, h // 180 // fct)
        k = 2 * r + 1
        eps = 0.0008 + 0.004 * s.soft_skin
        mean = g.box(small, k)
        var = g.sub(g.box(g.mul(small, small), k), g.mul(mean, mean))
        a = g.op("Div", [var, g.add(var, g.c(eps))])
        b = g.mul(mean, g.sub(g.c(1.0), a))
        a, b = g.box(a, k), g.box(b, k)
        # 피부 마스크 (filters._skin_mask: cv2 RGB2YCrCb float, Cr/Cb 중심 0.5)
        sc = g.clip01(small)
        yy = g.gray(sc)
        cr = g.mul(g.add(g.mul(g.sub(g.channel(sc, 0), yy), g.c(0.713)), g.c(0.5)), g.c(255.0))
        cb = g.mul(g.add(g.mul(g.sub(g.channel(sc, 2), yy), g.c(0.564)), g.c(0.5)), g.c(255.0))

        def ramp(x, lo, hi, up):
            return g.clip01(g.mul(g.sub(x, g.c(lo)) if up else g.sub(g.c(hi), x), g.c(1 / 8)))
        m = g.mul(g.mul(ramp(cr, 133, None, True), ramp(cr, None, 173, False)),
                  g.mul(ramp(cb, 77, None, True), ramp(cb, None, 127, False)))
        m = g.gauss(m, 2, ch=1)
        wgt = g.mul(g.add(g.mul(m, g.c(0.85)), g.c(0.15)), g.c(min(1.0, s.soft_skin)))
        img = g.affine(img, g.add(g.c(1.0), g.mul(wgt, g.sub(a, g.c(1.0)))), g.mul(wgt, b), h, w)
    if s.sharpen > 0:
        blur = g.gauss(img, max(0.8, h / 1440))
        img = g.weighted(img, 1.0 + s.sharpen, blur, -s.sharpen)
    clarity = getattr(s, "clarity", 0.0)
    if clarity > 0:
        small = g.area_down(img, 8)
        blur = g.resize(g.gauss(small, max(2.0, (h // 8) / 40)), h, w, "linear")
        img = g.weighted(img, 1.0 + 0.6 * clarity, blur, -0.6 * clarity)
    vib = getattr(s, "vibrance", 0.0)
    if vib > 0:
        gr = g.gray(img)
        # opset 17 의 ReduceMax/Min 은 축을 속성으로 받는다 (ReduceSum 은 입력)
        sat = g.sub(g.op("ReduceMax", [img], axes=[1], keepdims=1), g.op("ReduceMin", [img], axes=[1], keepdims=1))
        gain = g.add(g.c(1.0), g.mul(g.c(0.8 * vib), g.sub(g.c(1.0), g.clip01(sat))))
        img = g.add(gr, g.mul(g.sub(img, gr), gain))
    if s.glow > 0:
        small = g.area_down(img, 8)
        bright = g.clip01(g.mul(g.sub(small, g.c(0.55)), g.c(1 / 0.45)))
        gl = g.mul(g.gauss(bright, max(2.0, (h // 8) / 60)), g.c(0.6 * s.glow))
        img = g.affine(img, g.sub(g.c(1.0), gl), gl, h, w)
    if s.grain > 0:
        inputs.append(helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, 1, h // 2, w // 2]))
        img = g.add(img, g.mul(g.resize("noise", h, w, "linear"), g.c(0.025 * s.grain)))
    # filters.to_u8: max(img, 0) × 255 → 반올림·포화
    u = g.op("Round", [g.clip01(g.mul(img, g.c(255.0)), 0.0, 255.0)])
    if s.brighten > 0:
        # filters.brighten_u8: LUT(감마) → 회색과의 차이로 채도 조금 올림 (uint8 단계마다 반올림)
        lut = g.op("Floor", [g.add(g.mul(g.op("Pow", [g.mul(u, g.c(1 / 255)), g.c(1.0 - 0.25 * s.brighten)]),
                                         g.c(255.0)), g.c(0.5))])
        gr = g.op("Round", [g.gray(lut)])
        k = 0.2 * s.brighten
        u = g.op("Round", [g.clip01(g.weighted(lut, 1.0 + k, gr, -k), 0.0, 255.0)])
    # uint8 로 바꾼 뒤 NHWC 로 (float 상태로 전치하면 앞의 확대(Resize)와 묶여 GPU 에서 수십 배 느려졌다)
    u8 = g.op("Cast", [u], to=TensorProto.UINT8)
    g.nodes.append(helper.make_node("Transpose", [u8], ["out"], perm=[0, 2, 3, 1]))
    graph = helper.make_graph(g.nodes, "post", inputs,
                              [helper.make_tensor_value_info("out", TensorProto.UINT8, [1, oh, ow, 3])], g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


class GpuPost:
    def __init__(self, plan, s, device):
        self.plan, self.s, self.device = plan, s, device
        self._lock = threading.Lock()
        self._models = {}
        self._local = threading.local()

    def _compiled(self, y_shape):
        key = y_shape
        with self._lock:
            if key not in self._models:
                p = self.plan
                need_base = self.s.texture > 0 or y_shape is None
                xml = _build(self.s, y_shape, (p.rh, p.rw), p.oh, p.ow, need_base)
                core = backends._ov_core()
                # f16: f32 보다 약 40% 빠르고 결과 차이는 최대 2~3단계(0~255) — 실제 음악방송 프레임 실측
                self._models[key] = core.compile_model(core.read_model(xml), self.device.partition(":")[2],
                                                       {"PERFORMANCE_HINT": "THROUGHPUT",
                                                        "INFERENCE_PRECISION_HINT": "f16"})
            return self._models[key]

    def __call__(self, y, frame):
        """Upscaler.finish + filters.apply_all 과 같은 결과 (uint8, oh×ow×3)."""
        y_shape = None if y is None else y.shape[:2]
        compiled = self._compiled(y_shape)
        reqs = self._local.__dict__.setdefault("reqs", {})
        req = reqs.get(id(compiled))
        if req is None:
            req = reqs[id(compiled)] = compiled.create_infer_request()
        feed = {}
        names = {i.any_name for i in compiled.inputs}
        if "y" in names:
            feed["y"] = np.ascontiguousarray(y, np.float32)[None]
        if "frame" in names:
            feed["frame"] = np.ascontiguousarray(frame)[None]
        if "noise" in names:
            feed["noise"] = filters.grain_noise(self.plan.ow, self.plan.oh)[None, None]
        return req.infer(feed)["out"][0].copy()


def make(plan, s, device):
    """이 설정·장치에서 GPU 후처리를 쓸 수 있으면 GpuPost, 아니면 None (CPU 로 처리)."""
    if backends.ov is None or not device.startswith("ov:GPU"):
        return None
    oh, ow = plan.oh, plan.ow
    if s.soft_skin > 0 and (oh % 4 or ow % 4):
        return None
    if (getattr(s, "clarity", 0.0) > 0 or s.glow > 0) and (oh % 8 or ow % 8):
        return None
    return GpuPost(plan, s, device)
