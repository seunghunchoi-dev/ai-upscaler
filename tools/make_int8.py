"""AI 복원 모델(General v3) 의 INT8 경량 모델을 미리 만든다 (개발용 · NNCF 필요).

사용법: python tools/make_int8.py 보정용_영상_또는_사진 [...]
→ models/int8/general_v3_dn30.xml ... dn90.xml

압축 노이즈 제거 강도(denoise)마다 가중치가 달라서 5단계로 만들고, 실행할 때는 가장 가까운 단계를 쓴다.
보정 데이터는 활성값 범위 통계만 남기며 이미지는 모델에 저장되지 않는다.
첫·마지막 합성곱은 경량화하지 않는다 (화질에 민감, 속도 차이는 거의 없음).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nncf  # noqa: E402
import openvino as ov  # noqa: E402

from vup import backends, esrgan, video_io  # noqa: E402
from vup.paths import MODELS  # noqa: E402

W, H = 480, 272  # 보정은 작은 크기로 (모델은 크기와 무관하게 동작)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def samples(paths, per_video=32):
    out = []
    for p in paths:
        if Path(p).suffix.lower() in IMAGE_EXT:
            img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
            continue
        info = video_io.probe(p)
        for i in range(per_video):
            f = list(video_io.FrameReader(info, W, H, False, int(info.frames * (i + 0.5) / per_video), 1))
            if f:
                out.append(f[0])
    return [np.ascontiguousarray(x)[None] for x in out]


def main():
    import os
    data = samples(sys.argv[1:], per_video=int(os.environ.get("VUP_CALIB_FRAMES", "64")))
    print(f"보정 데이터 {len(data)}장")
    core = backends._ov_core()
    outdir = MODELS / "int8"
    outdir.mkdir(exist_ok=True)
    for level in esrgan.INT8_LEVELS:
        model = core.read_model(esrgan.build_onnx(level / 100, "general_v3"))
        convs = [op.get_friendly_name() for op in model.get_ops() if op.get_type_name() == "Convolution"]
        q = nncf.quantize(model, nncf.Dataset(data), subset_size=len(data), fast_bias_correction=True,
                          preset=nncf.QuantizationPreset.MIXED,
                          ignored_scope=nncf.IgnoredScope(names=[convs[0], convs[-1]]))
        path = outdir / f"general_v3_dn{level}.xml"
        ov.save_model(q, str(path))
        print("저장:", path, flush=True)


if __name__ == "__main__":
    main()
