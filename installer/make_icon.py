"""프로그램 아이콘(.ico) 생성: 화면의 로고와 같은 그림(vup.ui.APP_SVG)을 여러 크기로 그려 넣는다.
보라 그라데이션 둥근 사각형 + 반짝임(AI)."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from vup.ui import app_pixmap  # noqa: E402


def png_bytes(size):
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    app_pixmap(size).save(buf, "PNG")
    return bytes(buf.data())


def main(out):
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 (그림 그리기에 필요)
    sizes = [256, 64, 48, 32, 24, 16]
    pngs = [png_bytes(s) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, data = b"", b""
    for s, png in zip(sizes, pngs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    with open(out, "wb") as f:
        f.write(header + entries + data)
    print("아이콘 저장:", out)


if __name__ == "__main__":
    main(sys.argv[1])
