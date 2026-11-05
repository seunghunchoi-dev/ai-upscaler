"""명령줄 실행.

예) python -m vup 입력.mp4 --auto                       영상 분석 후 추천값으로 처리
    python -m vup 입력.mp4 --res 2160 --fps 60 --style 뽀샤시
    python -m vup 입력.mp4 --filter --res 2160               AI 없이 필터만 (매우 빠름)
    python -m vup 입력.mp4 --interp-only --fps 60            AI 화질 복원 없이 보간만
    python -m vup --devices                              사용 가능한 장치 보기
"""
import argparse
import sys
import time
from pathlib import Path

from . import backends, pipeline, video_io


def output_name(path, s, info=None, suffix="", ext=".mp4"):
    p = Path(path)
    if s.target_height:
        res = pipeline.resolution_name(s.target_height).split(" (")[0].replace(" ", "_")
    else:
        res = "원본크기" if s.scale == 1 else f"{s.scale:g}x"
    fps = ""
    if s.fps != "src":
        fps = f"_{float(pipeline.target_fps(info.fps, s.fps)):.0f}fps" if info else f"_{s.fps}fps"
    return p.with_name(f"{p.stem}_{res}{fps}{suffix}{ext}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vup", description="AI 영상 복원·업스케일·프레임 보간")
    ap.add_argument("input", nargs="?")
    ap.add_argument("-o", "--output")
    ap.add_argument("--auto", action="store_true", help="영상을 분석해 추천값을 먼저 적용")
    ap.add_argument("--quality", choices=list(pipeline.QUALITY_PRESETS), help="AI 처리 화질·속도 프리셋")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--filter", action="store_true",
                      help="AI 없이 필터만으로 처리 (매우 빠름 · 원본 프레임레이트, GUI '복원' 탭의 '필터만' 과 같음)")
    mode.add_argument("--interp-only", action="store_true",
                      help="AI 화질 복원 없이 단순 확대 + 보간만 (GUI '복원' 탭의 '보간만' 과 같음)")
    ap.add_argument("--style", choices=list(pipeline.STYLE_PRESETS))
    ap.add_argument("--res", type=int, help="출력 해상도(짧은 변): 480 720 1080 1440 2160 4320, 0=원본 크기")
    ap.add_argument("--scale", type=float, help="출력 배율 (--res 대신)")
    ap.add_argument("--fps", help="src | x2 | x3 | x4 | 60 | 59.94 ...")
    ap.add_argument("--denoise", type=float)
    ap.add_argument("--device", help="업스케일 장치 (--devices 로 확인)")
    ap.add_argument("--rife-device")
    ap.add_argument("--codec", choices=["h264", "h265", "av1"])
    ap.add_argument("--encoder", choices=["auto", "software", "nvenc", "qsv", "amf"])
    ap.add_argument("--audio", choices=["auto", "aac", "none"])
    ap.add_argument("--crf", type=int)
    ap.add_argument("--start", type=float, default=0.0, help="시작 위치(초)")
    ap.add_argument("--duration", type=float, help="처리할 길이(초) — 샘플용")
    ap.add_argument("--set", action="append", default=[], metavar="이름=값",
                    help="세부 설정 직접 지정 (예: --set glow=0.3)")
    ap.add_argument("--devices", action="store_true")
    a = ap.parse_args(argv)

    if a.devices:
        for d, name in backends.list_devices():
            print(f"{d:10s} {name}")
        print("자동 선택:", pipeline.resolve_devices(pipeline.Settings()))
        return 0
    if not a.input:
        ap.print_help()
        return 1

    info = video_io.probe(a.input)
    s = pipeline.Settings()
    if a.auto:
        from . import analyze
        r = analyze.recommend(info, s)
        for k, v in r.settings.items():
            setattr(s, k, v)
        print("영상 분석 결과:")
        for line in r.reasons:
            print("  -", line)
    if a.quality:
        for k, v in pipeline.QUALITY_PRESETS[a.quality].items():
            setattr(s, k, v)
    if a.style:
        for k, v in pipeline.STYLE_PRESETS[a.style].items():
            setattr(s, k, v)
    if a.filter:
        s.engine = "filter"
        for k, v in pipeline.FILTER_LOOK.items():
            setattr(s, k, v)
    if a.interp_only:
        s.ai_restore = False
    if a.res is not None:
        s.target_height, s.scale = a.res, 1.0
    if a.scale is not None:
        s.target_height, s.scale = 0, a.scale
    for k, attr in (("fps", "fps"), ("denoise", "denoise"), ("device", "device_sr"),
                    ("rife_device", "device_rife"), ("codec", "codec"), ("encoder", "encoder"),
                    ("audio", "audio"), ("crf", "crf")):
        if getattr(a, k) is not None:
            setattr(s, attr, getattr(a, k))
    for kv in a.set:
        k, _, v = kv.partition("=")
        cur = getattr(s, k)
        setattr(s, k, (v.lower() in ("1", "true", "on")) if isinstance(cur, bool) else type(cur)(v))

    if s.engine == "filter":
        s.fps = "src"  # 필터만은 보간 없이 원본 프레임레이트로 저장 (안내 문구·파일 이름도 맞춘다)
    out = a.output or str(output_name(a.input, s, info))
    plan = pipeline.Plan(info, s)
    print(f"{info.width}x{info.height} {info.fps_float:.2f}fps → {plan.ow}x{plan.oh} {float(plan.fout):.2f}fps")
    print("장치:", pipeline.resolve_devices(s))

    def show(p):
        bar = "#" * int(30 * p.done / max(1, p.total))
        m, sec = divmod(int(p.eta), 60)
        sys.stdout.write(f"\r[{bar:30s}] {p.done}/{p.total} {p.fps:5.2f}fps 남은 {m}:{sec:02d} {p.stage}   ")
        sys.stdout.flush()

    t = time.time()
    pipeline.process(a.input, out, s, show, start=a.start, duration=a.duration)
    print(f"\n완료: {out} ({time.time() - t:.0f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
