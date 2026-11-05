"""FFmpeg 를 파이프로 연결해 프레임을 읽고 쓴다 (디스크에 이미지 저장 없음)."""
import functools
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .paths import find_tool

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, creationflags=_NO_WINDOW, **kw)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: float
    interlaced: bool
    matrix: str          # 색 변환 행렬 (bt709 / bt470bg 등)
    audio_codec: str     # 없으면 ""
    video_codec: str = ""
    bitrate: int = 0     # 영상 비트레이트 (bps, 모르면 추정)

    @property
    def fps_float(self):
        return float(self.fps)


def probe(path):
    out = _run([find_tool("ffprobe"), "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", path])
    if out.returncode != 0:
        raise RuntimeError("영상 정보를 읽을 수 없습니다:\n" + out.stderr.decode("utf-8", "replace"))
    data = json.loads(out.stdout.decode("utf-8", "replace"))
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError("영상 스트림이 없습니다.")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    fps = Fraction(v.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(v.get("r_frame_rate") or "30/1")
    fps = fps.limit_denominator(1001)
    duration = float(v.get("duration") or data["format"].get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or int(round(duration * fps))

    w, h = int(v["width"]), int(v["height"])
    # 회전 메타데이터가 있으면 가로세로를 바꿔 준다
    for sd in v.get("side_data_list", []):
        if abs(int(sd.get("rotation", 0))) in (90, 270):
            w, h = h, w

    matrix = v.get("color_space") or ""
    if matrix in ("", "unknown", "reserved"):
        matrix = "bt709" if h >= 720 else "bt470bg"
    if matrix == "bt2020nc":
        matrix = "bt2020"
    if matrix not in ("bt709", "bt470bg", "smpte170m", "bt2020", "fcc", "smpte240m"):
        matrix = "bt709"

    # 영상 비트레이트: 스트림 값 → 전체 값에서 오디오 빼기 → 파일 크기로 추정 (webm/mkv 는 스트림 값이 없음)
    bitrate = int(v.get("bit_rate") or 0)
    if not bitrate:
        total = int(data["format"].get("bit_rate") or 0)
        if not total and duration:
            total = int(int(data["format"].get("size") or 0) * 8 / duration)
        audio_br = int((a or {}).get("bit_rate") or (128000 if a else 0))
        bitrate = max(0, total - audio_br)

    field = v.get("field_order", "progressive")
    return VideoInfo(path, w, h, fps, frames, duration,
                     field not in ("progressive", "unknown", ""), matrix,
                     a["codec_name"] if a else "", v.get("codec_name", ""), bitrate)


class FrameReader:
    """원본을 디코딩해 (h, w, 3) uint8 RGB 프레임을 순서대로 내보낸다."""

    def __init__(self, info, out_w, out_h, deinterlace=False, start_frame=0, max_frames=None):
        self.w, self.h = out_w, out_h
        vf = []
        if deinterlace:
            vf.append("bwdif=mode=send_frame:parity=auto:deint=all")
        vf.append(f"fps={info.fps.numerator}/{info.fps.denominator}")
        vf.append(f"scale={out_w}:{out_h}:flags=lanczos:in_color_matrix={info.matrix}:in_range=auto")
        vf.append("format=rgb24")
        args = [find_tool("ffmpeg"), "-v", "error", "-nostdin"]
        if start_frame:
            args += ["-ss", f"{float(start_frame / info.fps):.6f}"]
        args += ["-i", info.path, "-map", "0:v:0", "-an", "-sn", "-vf", ",".join(vf)]
        if max_frames:
            args += ["-frames:v", str(max_frames)]
        args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_NO_WINDOW, bufsize=out_w * out_h * 3 * 2)
        self._err = []
        threading.Thread(target=lambda: self._err.append(self.proc.stderr.read()), daemon=True).start()

    def __iter__(self):
        size = self.w * self.h * 3
        while True:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


# ── 인코더 ────────────────────────────────────────────────────────────────

CODECS = {  # 코덱 → 하드웨어 인코더 우선순위 (마지막은 소프트웨어)
    "h264": ["h264_nvenc", "h264_qsv", "h264_amf", "libx264"],
    "h265": ["hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"],
    "av1": ["av1_nvenc", "av1_qsv", "av1_amf", "libsvtav1"],
}


@functools.lru_cache(maxsize=None)
def encoder_works(name):
    """실제로 작은 영상을 인코딩해 보고 이 PC 에서 쓸 수 있는지 확인한다."""
    r = _run([find_tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
              "color=black:s=640x360:r=30", "-frames:v", "3", "-c:v", name,
              "-pix_fmt", "yuv420p", "-f", "null", "-"], timeout=30)
    return r.returncode == 0


ENCODER_KINDS = {"nvenc": "NVIDIA NVENC", "qsv": "Intel Quick Sync", "amf": "AMD AMF"}


def pick_encoder(codec, mode="auto"):
    """mode: auto(하드웨어 우선) | software | nvenc | qsv | amf.
    고른 하드웨어 인코더를 이 PC 에서 쓸 수 없으면 자동 선택으로 대체한다."""
    cands = CODECS[codec]
    if mode == "software":
        return cands[-1]
    if mode in ENCODER_KINDS:
        named = next(c for c in cands if c.endswith("_" + mode))
        if encoder_works(named):
            return named
    for c in cands:
        if encoder_works(c):
            return c
    return cands[-1]


def _audio_args(path, audio_codec, mode):
    mp4 = path.lower().endswith((".mp4", ".mov", ".m4v"))
    if mode == "aac" or (mp4 and audio_codec not in ("aac", "mp3", "ac3", "eac3", "alac")):
        return ["-c:a", "aac", "-b:a", "320k" if mode == "aac" else "256k"]
    return ["-c:a", "copy"]


def _quality_args(enc, crf):
    """crf: 0(최고)~51. 인코더마다 다른 옵션 이름으로 옮긴다."""
    if enc.endswith("_nvenc"):
        return ["-preset", "p6", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    if enc.endswith("_qsv"):
        return ["-preset", "slow", "-global_quality", str(crf)]
    if enc.endswith("_amf"):
        return ["-quality", "quality", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]
    if enc == "libsvtav1":
        return ["-preset", "6", "-crf", str(crf)]
    return ["-preset", "medium", "-crf", str(crf)]


class FrameWriter:
    """RGB 프레임을 받아 인코딩한다. audio_from 이 있으면 원본 오디오를 붙인다."""

    def __init__(self, path, w, h, fps, encoder, crf=20, ten_bit=False,
                 audio_from=None, audio_codec="", audio_start=0.0, audio_dur=None, audio_mode="auto"):
        if encoder.startswith("libx264") or encoder == "h264_amf":
            ten_bit = False  # 호환성을 위해 H.264 는 8비트만
        pix = ("p010le" if encoder.endswith(("_qsv", "_nvenc", "_amf")) else "yuv420p10le") \
            if ten_bit else ("nv12" if encoder.endswith("_qsv") else "yuv420p")
        args = [find_tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                "-r", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0"]
        if audio_from:
            if audio_start:
                args += ["-ss", f"{audio_start:.6f}"]
            if audio_dur:
                args += ["-t", f"{audio_dur:.6f}"]
            args += ["-i", audio_from]
        args += ["-map", "0:v:0"]
        if audio_from:
            args += ["-map", "1:a:0?", *_audio_args(path, audio_codec, audio_mode)]
        args += ["-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={pix}",
                 "-c:v", encoder, *_quality_args(encoder, crf),
                 "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                 "-color_range", "tv"]
        if encoder in ("libx265", "hevc_qsv", "hevc_nvenc", "hevc_amf"):
            args += ["-tag:v", "hvc1"]
        if path.lower().endswith((".mp4", ".mov", ".m4v")):
            # 시간 단위를 fps 에 맞춰 59.94/60fps 프레임 간격이 반올림 없이 정확하게 기록되도록
            args += ["-movflags", "+faststart", "-video_track_timescale", str(fps.numerator)]
        args.append(path)
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_NO_WINDOW)
        self._err = []
        threading.Thread(target=lambda: self._err.append(self.proc.stderr.read()), daemon=True).start()

    def write(self, frame):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError("인코더가 종료되었습니다:\n" + self.error())

    def error(self):
        self.proc.wait()
        return b"".join(self._err).decode("utf-8", "replace")

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError("인코딩 실패:\n" + self.error())

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def concat(parts, out_path, fps, audio_from=None, audio_codec="", audio_start=0.0, list_dir=None,
           audio_mode="auto"):
    """조각 영상들을 다시 인코딩하지 않고 이어 붙인 뒤 오디오를 합친다."""
    lst = list_dir / "concat.txt"
    lst.write_text("".join(f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                           for p in parts), encoding="utf-8")
    args = [find_tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio_from:
        if audio_start:
            args += ["-ss", f"{audio_start:.6f}"]
        args += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a:0?",
                 *_audio_args(out_path, audio_codec, audio_mode)]
    args += ["-c:v", "copy"]
    if out_path.lower().endswith((".mp4", ".mov", ".m4v")):
        args += ["-movflags", "+faststart", "-video_track_timescale", str(fps.numerator)]
    args.append(out_path)
    r = _run(args)
    if r.returncode != 0:
        raise RuntimeError("영상 합치기 실패:\n" + r.stderr.decode("utf-8", "replace"))
