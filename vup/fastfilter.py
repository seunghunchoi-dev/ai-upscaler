"""AI 없이 필터만으로 화질을 개선하는 빠른 처리 (FFmpeg 안에서 전부 처리).

  블록 완화 → 노이즈 제거 → CAS 선명화 → 명료도(입체감) → 생생함(vibrance) → 밝기·감마
  → 확대 (Intel 하드웨어 고품질 모드, 없으면 lanczos) → 하드웨어 인코딩

프레임을 Python 으로 옮기지 않아 실시간에 가깝게 빠르다. 없던 디테일을 만들지는 못하고(AI 만 가능),
있는 정보를 또렷하게 강조한다. AI 보간이 없으므로 원본 프레임레이트로 저장한다.
"""
import subprocess
import threading

import numpy as np

from . import video_io
from .paths import find_tool
from .video_io import _NO_WINDOW


def cpu_filters(s, info, deint):
    """원본 해상도에서 적용하는 필터 목록 (설정값 0 이면 그 필터는 빠짐)."""
    vf = []
    if deint:
        vf.append("bwdif=mode=send_frame:parity=auto:deint=all")
    d = float(s.denoise)
    if d >= 0.5:
        vf.append(f"deblock=filter={'strong' if d >= 0.7 else 'weak'}:block=8")
    if d > 0:
        ls = 1.0 + 3.0 * d
        vf.append(f"hqdn3d={ls:.2f}:{ls * 0.8:.2f}:{ls * 1.5:.2f}:{ls * 1.2:.2f}")
    if s.sharpen > 0:
        vf.append(f"cas=strength={min(1.0, 0.15 + 0.7 * s.sharpen):.2f}")
    if s.clarity > 0:  # 넓은 반경의 약한 선명화 = 명료도(입체감). unsharp 는 13x13 이 최대 (가로+세로 ≤ 25)
        vf.append(f"unsharp=luma_msize_x=13:luma_msize_y=13:luma_amount={1.2 * s.clarity:.2f}"
                  ":chroma_msize_x=3:chroma_msize_y=3:chroma_amount=0")
    if s.vibrance > 0:  # RGB 로 바꿀 때 색 변환 방식(BT.709)을 명시해 색이 틀어지지 않게
        vf += [f"scale=in_color_matrix={info.matrix}:out_color_matrix=bt709,format=gbrp",
               f"vibrance=intensity={0.6 * s.vibrance:.2f}",
               "scale=in_color_matrix=bt709:out_color_matrix=bt709,format=yuv420p"]
    if s.brighten > 0:
        vf.append(f"eq=gamma={1 + 0.25 * s.brighten:.3f}:saturation={1 + 0.1 * s.brighten:.3f}")
    return vf


def build_command(path, out_path, s, info, plan, encoder, start=0.0, duration=None):
    # 확대는 CPU(lanczos, 멀티코어)로, 인코딩만 하드웨어로 한다.
    # 필터를 거친 프레임을 Intel GPU 로 올리는 방식(hwupload)은 PC·드라이버에 따라 실패해서
    # (이 PC 에서도 실패) 어디서나 동작하는 방식으로 고정. 실측: 영상 40초를 42초에 처리.
    vf = cpu_filters(s, info, plan.deint)
    args = [find_tool("ffmpeg"), "-v", "error", "-y", "-nostdin"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-i", path, "-map", "0:v:0"]
    if s.audio != "none" and info.audio_codec:
        args += ["-map", "0:a:0?", *video_io._audio_args(out_path, info.audio_codec, s.audio)]
    ten = s.ten_bit and encoder not in ("libx264", "h264_qsv", "h264_amf", "h264_nvenc")
    hw = encoder.endswith(("_qsv", "_nvenc", "_amf"))
    pix = ("p010le" if hw else "yuv420p10le") if ten else ("nv12" if encoder.endswith("_qsv") else "yuv420p")
    vf += [f"scale={plan.ow}:{plan.oh}:flags=lanczos", f"format={pix}"]
    args += ["-vf", ",".join(vf), "-c:v", encoder, *video_io._quality_args(encoder, s.crf),
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    if encoder in ("libx265", "hevc_qsv", "hevc_nvenc", "hevc_amf"):
        args += ["-tag:v", "hvc1"]
    if out_path.lower().endswith((".mp4", ".mov", ".m4v")):
        args += ["-movflags", "+faststart"]
    args += ["-progress", "pipe:1", "-nostats", out_path]
    return args


def process(path, out_path, s, info, plan, encoder, on_progress, stop_event, start=0.0, duration=None,
            progress_cls=None):
    """필터 엔진으로 변환. 진행률은 FFmpeg 의 -progress 출력에서 읽는다."""
    import time
    total = int((duration if duration else max(0.0, info.duration - start)) * info.fps_float)
    total = max(1, total)
    cmd = build_command(path, out_path, s, info, plan, encoder, start, duration)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
                            text=True, encoding="utf-8", errors="replace")
    err = []
    threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True).start()
    t0, done = time.time(), 0
    for line in proc.stdout:
        if stop_event.is_set():
            proc.kill()
            break
        if line.startswith("frame="):
            try:
                done = int(line.split("=", 1)[1])
            except ValueError:
                continue
            el = max(1e-6, time.time() - t0)
            fps = done / el
            on_progress(progress_cls(min(done, total), total, fps, (total - done) / fps if fps > 0 else 0, "처리 중"))
    code = proc.wait()
    if stop_event.is_set():
        return None
    if code != 0:
        raise RuntimeError("필터 처리 실패:\n" + "".join(err))
    on_progress(progress_cls(total, total, 0, 0, "완료"))
    return out_path


def preview(path, s, info, plan, time_sec):
    """한 장면을 필터로 처리해 (원본 확대, 결과) RGB 이미지로 (미리보기는 lanczos 확대)."""
    def grab(vf):
        cmd = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{max(0.0, time_sec):.3f}", "-i", path,
               "-frames:v", "1", "-vf", ",".join(vf + [f"scale={plan.ow}:{plan.oh}:flags=lanczos:"
                                                      f"in_color_matrix={info.matrix}", "format=rgb24"]),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
        if r.returncode != 0 or len(r.stdout) < plan.ow * plan.oh * 3:
            raise RuntimeError("미리보기 실패:\n" + r.stderr.decode("utf-8", "replace"))
        return np.frombuffer(r.stdout[:plan.ow * plan.oh * 3], np.uint8).reshape(plan.oh, plan.ow, 3)
    return grab(["bwdif"] if plan.deint else []), grab(cpu_filters(s, info, plan.deint))
