"""연산 장치 백엔드.

같은 ONNX 모델을 아래 엔진 중 하나로 실행한다.
  ov:CPU / ov:GPU / ov:NPU   OpenVINO (Intel CPU, 내장/외장 Intel GPU, NPU)
  ort:cuda / ort:trt         ONNX Runtime + NVIDIA CUDA / TensorRT (onnxruntime-gpu 설치 시)
  ort:dml                    ONNX Runtime + DirectML (Windows 의 모든 GPU, onnxruntime-directml 설치 시)
  ort:cpu                    ONNX Runtime CPU
설치된 패키지에 따라 사용할 수 있는 장치만 목록에 나타난다.
"""
import functools

import numpy as np

from .paths import CACHE

try:
    import openvino as ov
except ImportError:
    ov = None

try:
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers() and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()  # pip 로 설치한 CUDA/cuDNN 라이브러리 불러오기
        except Exception:
            pass
except ImportError:
    ort = None

# NVIDIA GPU 를 여러 세션(업스케일·얼굴·보간)이 나눠 쓰므로 메모리를 필요한 만큼만 잡는다
_CUDA_OPTS = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "1"}


def _ort_providers(target):
    return {
        "trt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True,
                                               "trt_engine_cache_enable": True,
                                               "trt_engine_cache_path": str(CACHE)}),
                ("CUDAExecutionProvider", _CUDA_OPTS), "CPUExecutionProvider"],
        "cuda": [("CUDAExecutionProvider", _CUDA_OPTS), "CPUExecutionProvider"],
        "dml": [("DmlExecutionProvider", {"device_filter": "gpu",
                                          "performance_preference": "high_performance"}),
                "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[target]


_ORT_EP = {"trt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider",
           "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}


@functools.lru_cache(maxsize=None)
def _ort_works(target):
    """작은 모델로 세션을 만들어 실제로 그 장치에서 도는지 확인한다.
    onnxruntime-gpu 는 드라이버·CUDA·cuDNN·TensorRT 가 없어도 목록에는 장치를 올리고
    세션을 만들 때 조용히 CPU 로 넘어가므로, 목록만 보고 고르면 매우 느려진다."""
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "probe",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    so = ort.SessionOptions()
    so.log_severity_level = 4
    try:
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=_ort_providers(target))
        sess.run(None, {"x": np.ones((1, 4), np.float32)})
        return sess.get_providers()[0] == _ORT_EP[target]
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _ov_core():
    core = ov.Core()
    CACHE.mkdir(exist_ok=True)
    core.set_property({"CACHE_DIR": str(CACHE)})
    return core


@functools.lru_cache(maxsize=1)
def gpu_names():
    """Windows 에 설치된 그래픽 장치 이름 (레지스트리에서 빠르게 읽음)."""
    names = []
    try:
        import winreg
        key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                try:
                    with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sub:
                        names.append(winreg.QueryValueEx(sub, "DriverDesc")[0])
                except OSError:
                    pass
    except Exception:
        pass
    return names


def _other_gpu():
    """NVIDIA / AMD GPU 이름 (없으면 None)."""
    return next((n for n in gpu_names() if any(v in n.upper() for v in ("NVIDIA", "AMD", "RADEON"))), None)


@functools.lru_cache(maxsize=1)
def list_devices():
    """[(장치 id, 표시 이름)] — 빠른 장치가 앞에 오도록 정렬."""
    devs = []
    if ort is not None:
        prov = ort.get_available_providers()
        nv = next((n for n in gpu_names() if "NVIDIA" in n.upper()), "NVIDIA GPU")
        # CUDA 를 자동 선택(맨 앞)으로 둔다. TensorRT 는 해상도마다 엔진을 새로 만드느라
        # 처음 몇 분씩 멈춘 것처럼 보이므로 직접 고를 때만 쓴다.
        if "CUDAExecutionProvider" in prov and _ort_works("cuda"):
            devs.append(("ort:cuda", f"{nv} (CUDA)"))
            if "TensorrtExecutionProvider" in prov and _ort_works("trt"):
                devs.append(("ort:trt", f"{nv} (TensorRT, 첫 실행 느림)"))
    ov_gpus = []
    if ov is not None:
        core = _ov_core()
        for d in core.available_devices:
            try:
                name = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                name = d
            if d.startswith("GPU"):
                kind = "외장" if "dGPU" in str(core.get_property(d, "DEVICE_TYPE")) else "내장"
                ov_gpus.append((f"ov:{d}", f"Intel {kind} GPU - {name} (OpenVINO)", kind))
        ov_gpus.sort(key=lambda g: g[2] != "외장")
    devs += [(i, n) for i, n, kind in ov_gpus if kind == "외장"]
    # DirectML: NVIDIA/AMD GPU 가 있으면 Intel 내장 GPU 보다 앞에, 없으면 뒤에 둔다
    # (Intel 내장 GPU 만 있을 때는 OpenVINO 가 DirectML 보다 2배 이상 빠름)
    dml = None
    if ort is not None and "DmlExecutionProvider" in ort.get_available_providers():
        other = _other_gpu()
        dml = ("ort:dml", f"{other or 'GPU'} (DirectML)")
        if other:
            devs.append(dml)
            dml = None
    devs += [(i, n) for i, n, kind in ov_gpus if kind == "내장"]
    if dml:
        devs.append(dml)
    if ov is not None:
        core = _ov_core()
        if "CPU" in core.available_devices:
            devs.append(("ov:CPU", "CPU (OpenVINO)"))
        if "NPU" in core.available_devices:
            devs.append(("ov:NPU", "Intel NPU (OpenVINO, 실험적)"))
    if ort is not None:
        devs.append(("ort:cpu", "CPU (ONNX Runtime)"))
    if not devs:
        raise RuntimeError("openvino 또는 onnxruntime 이 필요합니다.")
    return devs


def auto_device():
    return list_devices()[0][0]


def device_label(dev_id):
    return dict(list_devices()).get(dev_id, dev_id)


class Runner:
    """ONNX 모델(바이트 또는 경로)을 고정 입력 크기로 컴파일해 실행한다."""

    def __init__(self, model, device, input_shape, fp16=True):
        if device == "auto" or device not in dict(list_devices()):
            # 다른 PC 에서 저장한 설정(예: CUDA)이 이 PC 에 없으면 자동 선택으로
            device = auto_device()
        self.device = device
        if not isinstance(model, (str, bytes)):
            model = str(model)
        engine, _, target = device.partition(":")
        if engine == "ov":
            self._init_ov(model, target, input_shape, fp16)
        else:
            self._init_ort(model, target)

    def _init_ov(self, model, target, input_shape, fp16):
        core = _ov_core()
        m = core.read_model(model)
        m.reshape({m.inputs[0].any_name: list(input_shape)})
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if target != "CPU":
            cfg["INFERENCE_PRECISION_HINT"] = "f16" if fp16 else "f32"
        compiled = core.compile_model(m, target, cfg)
        req = compiled.create_infer_request()
        out = compiled.outputs[0]
        self._run = lambda x: req.infer({0: x})[out]

    def _init_ort(self, model, target):
        so = ort.SessionOptions()
        if target == "dml":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(model, so, providers=_ort_providers(target))
        if sess.get_providers()[0] != _ORT_EP[target]:
            # 큰 모델에서만 실패하는 경우(GPU 메모리 부족 등): CPU 로 조용히 돌면 수십 배 느려지므로 알린다
            raise RuntimeError(f"{device_label(self.device)} 에서 모델을 불러오지 못했습니다 "
                               "(GPU 메모리 부족 또는 드라이버 문제). 처리 해상도를 낮추거나 다른 장치를 고르세요.")
        name = sess.get_inputs()[0].name
        self._run = lambda x: sess.run(None, {name: x})[0]

    def __call__(self, x):
        return np.asarray(self._run(np.ascontiguousarray(x)))
