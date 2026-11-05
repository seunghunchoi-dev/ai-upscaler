"""Real-ESRGAN 계열 업스케일 모델의 ONNX 그래프를 가중치(.npz)로부터 만든다.

- general_v3 : SRVGGNetCompact (120만 매개변수, 빠름). 두 가중치(강한/약한 노이즈 제거)를
               denoise 비율로 섞어(DNI) 실행 시점에 만든다.
- x4plus / x2plus : RRDBNet (1,670만 매개변수, 디테일 복원력이 훨씬 좋지만 느림).

모든 모델은 입력 uint8 NHWC(RGB) → 출력 float NHWC(0~1) 로 통일한다.
(형식 변환까지 모델 안에서 하므로 GPU 가 처리한다)
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .paths import MODELS

STRONG = MODELS / "realesr-general-x4v3.npz"
WEAK = MODELS / "realesr-general-wdn-x4v3.npz"

SR_MODELS = {
    "general_v3": dict(label="Real-ESRGAN General v3 (빠름)", scale=4, file=STRONG.name),
    "x4plus": dict(label="Real-ESRGAN x4plus (고품질 · 느림)", scale=4, file="RealESRGAN_x4plus.npz"),
    "x2plus": dict(label="Real-ESRGAN x2plus (고품질 · 2배)", scale=2, file="RealESRGAN_x2plus.npz"),
}


def model_scale(name):
    return SR_MODELS[name]["scale"]


# INT8 경량 모델 (tools/make_int8.py 로 미리 만듦): 압축 노이즈 제거 강도별 4단계.
# 30 단계는 경량화하면 파란색이 약간 빠져서(B −4) 제외 — 45 미만 설정은 45 단계로 처리된다.
INT8_LEVELS = (45, 60, 75, 90)


def int8_path(denoise):
    """denoise 와 가장 가까운 단계의 INT8 경량 모델 경로 (없으면 None)."""
    level = min(INT8_LEVELS, key=lambda lv: abs(lv - denoise * 100))
    p = MODELS / "int8" / f"general_v3_dn{level}.xml"
    return p if p.exists() else None


def int8_available():
    return any((MODELS / "int8" / f"general_v3_dn{lv}.xml").exists() for lv in INT8_LEVELS)


def available_models():
    return {k: v for k, v in SR_MODELS.items() if (MODELS / v["file"]).exists()}


class _Graph:
    def __init__(self):
        self.nodes, self.inits, self._n = [], [], 0

    def name(self, prefix):
        self._n += 1
        return f"{prefix}_{self._n}"

    def const(self, arr, prefix="c"):
        n = self.name(prefix)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), n))
        return n

    def op(self, op_type, inputs, prefix=None, **attrs):
        out = self.name(prefix or op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def conv(self, x, w, b):
        return self.op("Conv", [x, self.const(w, "w"), self.const(b, "b")],
                       kernel_shape=[w.shape[2], w.shape[3]], pads=[1, 1, 1, 1])

    def lrelu(self, x):
        return self.op("LeakyRelu", [x], alpha=0.2)

    def up_nearest(self, x, s):
        return self.op("Resize", [x, "", self.const(np.array([1, 1, s, s], np.float32))], mode="nearest",
                       coordinate_transformation_mode="asymmetric", nearest_mode="floor")

    def input_u8(self):
        """uint8 NHWC → float NCHW 0~1."""
        x = self.op("Cast", ["input"], to=TensorProto.FLOAT)
        x = self.op("Mul", [x, self.const(np.array(1 / 255, np.float32))])
        return self.op("Transpose", [x], perm=[0, 3, 1, 2])

    def finish(self, x, name):
        x = self.op("Clip", [x, self.const(np.array(0, np.float32)), self.const(np.array(1, np.float32))])
        self.nodes.append(helper.make_node("Transpose", [x], ["output"], perm=[0, 2, 3, 1]))
        graph = helper.make_graph(
            self.nodes, name,
            [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, "h", "w", 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "H", "W", 3])],
            self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model.SerializeToString()


def _blend(denoise):
    """denoise 0.0(디테일 보존) ~ 1.0(강한 압축노이즈 제거)."""
    denoise = float(np.clip(denoise, 0.0, 1.0))
    with np.load(STRONG) as s, np.load(WEAK) as w:
        return {k: (s[k] * denoise + w[k] * (1.0 - denoise)).astype(np.float32) for k in s.files}


def _build_srvgg(denoise):
    sd = _blend(denoise)
    g = _Graph()
    x0 = g.input_u8()
    x = x0
    n_layers = max(int(k.split(".")[1]) for k in sd) + 1
    i = 0
    while i < n_layers:
        x = g.conv(x, sd[f"body.{i}.weight"], sd[f"body.{i}.bias"])
        nxt = sd.get(f"body.{i + 1}.weight")
        if nxt is not None and nxt.ndim == 1:  # PReLU
            x = g.op("PRelu", [x, g.const(nxt.reshape(-1, 1, 1))])
            i += 2
        else:
            i += 1
    x = g.op("DepthToSpace", [x], blocksize=4, mode="CRD")
    x = g.op("Add", [x, g.up_nearest(x0, 4)])
    return g.finish(x, "srvgg")


def _build_rrdb(path, scale):
    with np.load(path) as z:
        sd = {k: z[k] for k in z.files}
    g = _Graph()
    x = g.input_u8()
    w_first = sd["conv_first.weight"]
    if scale in (1, 2):
        # PyTorch pixel_unshuffle 의 채널 순서는 c*r*r + (i*r+j), ONNX SpaceToDepth 는 (i*r+j)*C + c.
        # 연산을 추가하는 대신 첫 conv 가중치의 입력 채널 순서를 SpaceToDepth 에 맞게 바꾼다.
        r = 4 // scale
        x = g.op("SpaceToDepth", [x], blocksize=r)
        perm = [c * r * r + b for b in range(r * r) for c in range(3)]
        w_first = w_first[:, perm]
    feat = g.conv(x, w_first, sd["conv_first.bias"])
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("body.")})
    body = feat
    for bi in range(n_blocks):
        rrdb_in = body
        out = body
        for ri in (1, 2, 3):
            p = f"body.{bi}.rdb{ri}"
            rdb_in = out
            feats = [rdb_in]
            for ci in range(1, 6):
                inp = feats[0] if len(feats) == 1 else g.op("Concat", feats, axis=1)
                y = g.conv(inp, sd[f"{p}.conv{ci}.weight"], sd[f"{p}.conv{ci}.bias"])
                if ci < 5:
                    feats.append(g.lrelu(y))
            # RDB 출력 = conv5 × 0.2 + 입력
            out = g.op("Add", [g.op("Mul", [y, g.const(np.array(0.2, np.float32))]), rdb_in])
        body = g.op("Add", [g.op("Mul", [out, g.const(np.array(0.2, np.float32))]), rrdb_in])
    body = g.conv(body, sd["conv_body.weight"], sd["conv_body.bias"])
    feat = g.op("Add", [feat, body])
    feat = g.lrelu(g.conv(g.up_nearest(feat, 2), sd["conv_up1.weight"], sd["conv_up1.bias"]))
    feat = g.lrelu(g.conv(g.up_nearest(feat, 2), sd["conv_up2.weight"], sd["conv_up2.bias"]))
    feat = g.lrelu(g.conv(feat, sd["conv_hr.weight"], sd["conv_hr.bias"]))
    out = g.conv(feat, sd["conv_last.weight"], sd["conv_last.bias"])
    return g.finish(out, "rrdb")


def build_onnx(denoise=0.5, model="general_v3"):
    if model == "general_v3":
        return _build_srvgg(denoise)
    info = SR_MODELS[model]
    return _build_rrdb(MODELS / info["file"], info["scale"])
