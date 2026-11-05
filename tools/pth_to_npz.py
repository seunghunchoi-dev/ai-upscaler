"""PyTorch .pth 체크포인트를 torch 없이 읽어 .npz 로 변환한다.

사용법: python tools/pth_to_npz.py 입력.pth 출력.npz

pickle 은 허용된 이름만 해석하는 제한된 Unpickler 로 읽으므로
임의 코드가 실행되지 않는다.
"""
import pickle
import sys
import zipfile
from collections import OrderedDict

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor_v2(storage, offset, size, stride, *_):
    arr = storage
    itemsize = arr.itemsize
    return np.lib.stride_tricks.as_strided(
        arr[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    ).copy()


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf = zf
        self.prefix = prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"허용되지 않은 객체: {module}.{name}")

    def persistent_load(self, pid):
        _, storage_type, key, _location, _numel = pid
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=storage_type.dtype)


def load_pth(path):
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        with zf.open(pkl) as f:
            obj = _Unpickler(f, zf, prefix).load()
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    return obj


if __name__ == "__main__":
    sd = load_pth(sys.argv[1])
    np.savez(sys.argv[2], **{k: v.astype(np.float32) for k, v in sd.items()})
    for k, v in list(sd.items())[:4]:
        print(k, v.shape)
    print(f"... 총 {len(sd)}개 텐서 -> {sys.argv[2]}")
