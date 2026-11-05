"""PySide6 화면.

- 모든 옵션은 탭(기본 / 복원 / 보간 / 스타일 / 출력)에서 직접 고를 수 있다.
- 영상을 추가하면 분석해서 추천값을 자동으로 넣는다 (끌 수 있음).
- 설정은 파일마다 따로 저장된다. [추천값 적용] 은 '분석 · 추천' 칸에, [기본값] [모든 파일에 적용] 은 설정 아래에.
- 같은 기능은 한 곳에만 둔다: 처리 방식(AI 복원 · 보간만 · 필터만)과 품질/속도 프리셋은 '복원' 탭,
  스타일 프리셋은 '스타일' 탭, 출력 해상도·프레임레이트·연산 장치는 '기본' 탭.
"""
import faulthandler
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from . import __version__, analyze, backends, esrgan, pipeline, ui, video_io
from . import face as face_mod
from . import photo as photo_mod
from .cli import output_name
from .paths import CACHE, CONFIG, PRESETS, ensure_dirs
from .ui import DropList, FeatureCard, Segmented, Toggle, button, form_layout, icon, note

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv", ".wmv", ".mpg", ".mpeg")
RES_ITEMS = [("원본 크기 유지 (복원만)", 0)] + pipeline.RESOLUTIONS + [("직접 입력", -1)]
FPS_ITEMS = [("원본 그대로 (보간 안 함)", "src"), ("2배", "x2"), ("3배", "x3"), ("4배", "x4"),
             ("24fps", "24"), ("30fps", "30"), ("48fps", "48"), ("50fps", "50"), ("60fps", "60"),
             ("120fps", "120"), ("직접 입력", "custom")]
PROC_ITEMS = [("33% (가장 빠름 · 작은 얼굴·디테일 손실 큼)", 0.33), ("40% (빠름 · 디테일 손실 있음)", 0.4),
              ("50% (빠름 · 압축 깨짐에 강함)", 0.5), ("75% (균형)", 0.75), ("100% (디테일 최대 · 느림)", 1.0)]
ENCODER_ITEMS = [("자동 (하드웨어 우선)", "auto"), ("소프트웨어 (화질 최고 · 느림)", "software"),
                 ("NVIDIA NVENC", "nvenc"), ("Intel Quick Sync", "qsv"), ("AMD AMF", "amf")]
AUDIO_ITEMS = [("원본 그대로 (필요하면 AAC 변환)", "auto"), ("AAC 320kbps 로 변환", "aac"), ("소리 없음", "none")]
CUSTOM = "사용자 지정"
# 출력 해상도·프레임레이트: 품질/속도 프리셋과 따로 정하는 항목 (바꿔도 프리셋 표시가 바뀌지 않음)
OUTPUT_KEYS = {"target_height", "scale", "fps"}

ROLE_PATH, ROLE_INFO, ROLE_SET, ROLE_RECO, ROLE_WHY = (Qt.ItemDataRole.UserRole + i for i in range(5))

# 추천값 적용 시 "바뀐 항목" 표시용 이름과 값 표시 방법
_FIELD_NAMES = {
    "target_height": ("출력 해상도", lambda v: pipeline.resolution_name(v) if v else "원본 크기"),
    "fps": ("프레임레이트", lambda v: "원본" if v == "src" else f"{v}fps" if not str(v).startswith("x") else f"{v[1:]}배"),
    "proc_scale": ("처리 해상도", lambda v: f"{float(v) * 100:.0f}%"),
    "denoise": ("압축 노이즈 제거", lambda v: f"{float(v) * 100:.0f}"),
    "texture": ("원본 질감 유지", lambda v: f"{float(v) * 100:.0f}"),
    "face_enhance": ("얼굴 복원", lambda v: "켬" if v else "끔"),
    "scene_threshold": ("장면 전환 감지", lambda v: "끔" if float(v) >= 1 else f"{(0.45 - float(v)) / 0.4 * 100:.0f}"),
    "dedup": ("중복 프레임 보정", lambda v: "켬" if v else "끔"),
    "rife_model": ("보간 모델", lambda v: pipeline.RIFE_MODELS.get(v, v)),
    "deinterlace": ("인터레이스 제거", lambda v: {"auto": "자동", "on": "켬", "off": "끔"}.get(v, v)),
    "codec": ("코덱", lambda v: str(v).upper()),
    "crf": ("화질(CRF)", str),
    "ten_bit": ("10비트", lambda v: "켬" if v else "끔"),
}


def _describe_changes(old, new):
    out = []
    for k, (name, fmt) in _FIELD_NAMES.items():
        a, b = old.get(k), new.get(k)
        same = a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-6)
        if k in new and not same:
            out.append(f"{name} {fmt(a) if a is not None else '-'} → {fmt(b)}")
    return out

log = logging.getLogger("vup")


# ── 비교 미리보기 위젯 ─────────────────────────────────────────────────────

def _qimage(arr):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class CompareView(QWidget):
    """원본/결과를 세로선으로 나눠 비교. 왼쪽 드래그: 경계 이동, 휠: 확대, 오른쪽 드래그: 이동,
    더블클릭: 원래 크기."""

    def __init__(self):
        super().__init__()
        self.before = self.after = None
        self.split = 0.5
        self.zoom, self.center = 1.0, QPointF(0.5, 0.5)
        self._pan = None
        self.message = "영상을 추가하고 [미리보기] 를 눌러 보세요"
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip("왼쪽 드래그: 비교선 이동 · 휠: 확대 · 오른쪽 드래그: 이동 · 더블클릭: 원래대로")

    @Slot(object, object)
    def set_images(self, before, after):
        self.before = _qimage(before) if before is not None else None
        self.after = _qimage(after) if after is not None else None
        self.update()

    def set_message(self, text):
        self.message = text
        self.update()

    def _img(self):
        return self.after or self.before

    def _src_rect(self):
        img = self._img()
        w, h = img.width() / self.zoom, img.height() / self.zoom
        x = min(max(0, self.center.x() * img.width() - w / 2), img.width() - w)
        y = min(max(0, self.center.y() * img.height() - h / 2), img.height() - h)
        return QRectF(x, y, w, h)

    def _dst_rect(self, src):
        s = min(self.width() / src.width(), self.height() / src.height())
        w, h = src.width() * s, src.height() * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(ui.BORDER), 1))
        p.setBrush(QColor("#0a0c10"))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
        img = self._img()
        if img is None:
            c = QPointF(self.width() / 2, self.height() / 2)
            p.drawPixmap(QPointF(c.x() - 18, c.y() - 46), ui.pixmap("eye", ui.FAINT, 36))
            p.setPen(QColor(ui.MUTED))
            p.drawText(QRectF(0, c.y() + 2, self.width(), 30), Qt.AlignmentFlag.AlignHCenter, self.message)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2)
        src = self._src_rect()
        dst = self._dst_rect(src)
        if self.before is not None and self.after is not None:
            sx = dst.left() + dst.width() * self.split
            for image, clip in ((self.before, QRectF(dst.left(), dst.top(), sx - dst.left(), dst.height())),
                                (self.after, QRectF(sx, dst.top(), dst.right() - sx, dst.height()))):
                p.save()
                p.setClipRect(clip)
                p.drawImage(dst, image, src)
                p.restore()
            p.setPen(QPen(QColor(255, 255, 255, 220), 2))
            p.drawLine(QPointF(sx, dst.top()), QPointF(sx, dst.bottom()))
            # 비교선 손잡이
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPointF(sx, dst.center().y()), 11, 11)
            p.setPen(QPen(QColor(ui.BG), 2))
            y0 = dst.center().y()
            for d in (-1, 1):
                p.drawLine(QPointF(sx + d * 3, y0 - 4), QPointF(sx + d * 7, y0))
                p.drawLine(QPointF(sx + d * 7, y0), QPointF(sx + d * 3, y0 + 4))
            p.setFont(QFont(self.font().family(), 9, QFont.Weight.DemiBold))
            for text, x in (("원본", dst.left() + 12), ("결과", dst.right() - 12 - 56)):
                r = QRectF(x, dst.top() + 12, 56, 26)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(10, 12, 16, 170))
                p.drawRoundedRect(r, 13, 13)
                p.setPen(QColor(255, 255, 255))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        else:
            p.drawImage(dst, img, src)
        if self.zoom > 1:
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(dst.left() + 10, dst.bottom() - 30, 120, 24),
                       Qt.AlignmentFlag.AlignLeft, f"확대 {self.zoom:.1f}x")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._move_split(e.position())
        else:
            self._pan = (e.position(), QPointF(self.center))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._move_split(e.position())
        elif self._pan and self._img() is not None:
            start, c0 = self._pan
            dst = self._dst_rect(self._src_rect())
            d = e.position() - start
            self.center = QPointF(c0.x() - d.x() / dst.width() / self.zoom,
                                  c0.y() - d.y() / dst.height() / self.zoom)
            self.update()

    def mouseReleaseEvent(self, _):
        self._pan = None

    def _move_split(self, pos):
        if self._img() is None:
            return
        dst = self._dst_rect(self._src_rect())
        self.split = min(1, max(0, (pos.x() - dst.left()) / max(1, dst.width())))
        self.update()

    def wheelEvent(self, e):
        if self._img() is None:
            return
        self.zoom = min(8.0, max(1.0, self.zoom * (1.25 if e.angleDelta().y() > 0 else 0.8)))
        self.update()

    def mouseDoubleClickEvent(self, _):
        self.zoom, self.center = 1.0, QPointF(0.5, 0.5)
        self.update()


# ── 작은 위젯들 ────────────────────────────────────────────────────────────

class Slider(QWidget):
    changed = Signal()

    def __init__(self, tip=""):
        super().__init__()
        self.s = QSlider(Qt.Orientation.Horizontal)
        self.s.setRange(0, 100)
        self.lbl = QLabel()
        self.lbl.setFixedWidth(30)
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.s)
        lay.addWidget(self.lbl)
        self.setToolTip(tip)
        self.s.valueChanged.connect(self._on)
        self._on()

    def _on(self):
        self.lbl.setText(str(self.s.value()))
        self.changed.emit()

    def value(self):
        return self.s.value() / 100

    def setValue(self, v):
        self.s.blockSignals(True)
        self.s.setValue(int(round(float(v) * 100)))
        self.s.blockSignals(False)
        self.lbl.setText(str(self.s.value()))


def _combo(items, tip=""):
    c = QComboBox()
    # 긴 항목 이름 때문에 왼쪽 패널이 넓어지지 않도록 (펼치면 전체 이름이 보임)
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(8)
    for label, data in items:
        c.addItem(label, data)
    c.setToolTip(tip)
    return c


def _set_combo(c, data):
    for j in range(c.count()):
        d = c.itemData(j)
        if d == data or (isinstance(d, float) and isinstance(data, (int, float)) and abs(d - data) < 1e-6):
            c.setCurrentIndex(j)
            return True
    return False


# ── 백그라운드 작업 ────────────────────────────────────────────────────────

class Worker(QObject):
    progress = Signal(object, int, int)      # Progress, 현재 파일 번호, 전체 파일 수
    preview = Signal(object, object)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(self) or "")
        except pipeline.Stopped:
            self.finished.emit("중지됨")
        except Exception as e:
            log.error("작업 실패\n%s", traceback.format_exc())
            self.failed.emit(f"{e}\n\n(자세한 내용: cache\\error.log)")


# ── 사진 탭 ────────────────────────────────────────────────────────────────

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
PHOTO_SIZE_ITEMS = [("원본 크기 (복원만)", "x1"), ("2배", "x2"), ("4배", "x4"),
                    ("긴 변 FHD 1920px", 1920), ("긴 변 QHD 2560px", 2560), ("긴 변 4K 3840px", 3840),
                    ("긴 변 8K 7680px", 7680), ("직접 입력 (긴 변)", "custom")]
PHOTO_QUALITY_ITEMS = [("표준 (General v3 · 빠름)", "general_v3"),
                       ("고품질 (x4plus · 느림, 작은 사진에 추천)", "x4plus")]
PHOTO_FORMAT_ITEMS = [("원본과 같게", "same"), ("JPG", ".jpg"), ("PNG (무손실)", ".png"), ("WEBP", ".webp")]
_STATE_STYLE = {"자동": "background:#173b2c;color:#7ee2b0", "수정됨": "background:#3d3115;color:#ffd27a",
                "기본": "background:#262b38;color:#aab2c5", "분석 중": "background:#22305a;color:#a9b8ff"}


def load_image_rgb(path):
    """사진을 RGB numpy 로 읽는다. 휴대폰 사진의 회전 정보(EXIF)를 반영하고 한글 경로도 지원."""
    from PySide6.QtGui import QImageReader
    r = QImageReader(path)
    r.setAutoTransform(True)
    img = r.read()
    if img.isNull():
        raise RuntimeError(f"사진을 열 수 없습니다: {r.errorString()}")
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, bpl * h).reshape(h, bpl)[:, :w * 3]
    return arr.reshape(h, w, 3).copy()


class PhotoItem(QWidget):
    """사진 목록의 한 줄: 썸네일 · 이름 · 크기 · 설정 상태 · 진행 막대."""

    def __init__(self, thumb, name):
        super().__init__()
        from PySide6.QtGui import QPixmap
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 8, 4)
        pic = QLabel()
        pic.setFixedSize(64, 64)
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pic.setPixmap(QPixmap.fromImage(thumb))
        pic.setStyleSheet("background:#0a0c10;border-radius:8px")
        lay.addWidget(pic)
        col = QVBoxLayout()
        col.setSpacing(3)
        top = QHBoxLayout()
        self.name = QLabel(name)
        self.name.setStyleSheet("font-weight:600")
        self.name.setMinimumWidth(60)
        self.state = QLabel()
        self.state.setFixedHeight(20)
        top.addWidget(self.name, 1)
        top.addWidget(self.state)
        col.addLayout(top)
        self.detail = QLabel()
        self.detail.setObjectName("muted")
        col.addWidget(self.detail)
        prow = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        # 0% 일 때도 막대가 보이도록 빈 부분에 바탕색을 넣는다
        self.bar.setStyleSheet(f"QProgressBar{{background:#232836;border:none;border-radius:3px;min-height:6px}}"
                               f"QProgressBar::chunk{{background:{ui.ACCENT};border-radius:3px}}")
        self.pct = QLabel("대기")
        self.pct.setFixedWidth(64)
        self.pct.setObjectName("muted")
        prow.addWidget(self.bar, 1)
        prow.addWidget(self.pct)
        col.addLayout(prow)
        lay.addLayout(col, 1)
        self.set_state("기본")

    def set_state(self, kind):
        self.state.setText(f" {kind} ")
        self.state.setStyleSheet(f"border-radius:10px;font-size:8.5pt;padding:0 6px;{_STATE_STYLE[kind]}")

    def set_progress(self, value, text=None):
        self.bar.setValue(int(value))
        self.pct.setText(text if text is not None else f"{int(value)}%")


class PhotoPanel(QWidget):
    """사진 개선 탭.

    - 설정은 사진마다 따로 저장된다. '일괄' 모드에서는 바꾼 항목이 모든 사진에, '개별' 모드에서는
      선택한 사진에만 적용된다. (일괄 모드에서도 바꾸지 않은 항목은 사진별 자동값이 유지됨)
    - [자동 설정] 은 사진을 분석해 추천값으로 되돌린다. 추가할 때 자동 설정할 수도 있다.
    """

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._loading = False
        self._shown = None          # 지금 화면에 표시된 설정 (바뀐 항목을 찾는 기준)
        self.thread = self.worker = None
        self._on_done = None
        self._cache = {}            # (경로, 설정) → 미리보기 결과
        self.stop_event = threading.Event()
        split = QSplitter()
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left.setWidget(self._build_left())
        left.setMinimumWidth(460)
        split.addWidget(left)
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([490, 910])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self._show_settings(photo_mod.PhotoSettings())
        self._update_scope()

    # ── 왼쪽 ──
    def _build_left(self):
        from PySide6.QtWidgets import QAbstractItemView
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 6, 8)

        files = QGroupBox("사진 파일")
        fl = QVBoxLayout(files)
        self.files = DropList("사진이나 폴더를 여기에 끌어다 놓으세요")
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.files.setMinimumHeight(180)
        self.files.setMaximumHeight(300)
        self.files.currentRowChanged.connect(self._on_row_changed)
        self.files.itemSelectionChanged.connect(self._update_scope)
        fl.addWidget(self.files)
        row = QHBoxLayout()
        for text, ic, fn in (("추가", "plus", self._add_dialog), ("삭제", "minus", self._remove_selected),
                             ("모두 지우기", "trash", self._clear)):
            b = button(text, ic)
            b.clicked.connect(fn)
            row.addWidget(b)
        fl.addLayout(row)
        self.auto_on_add = Toggle("사진을 추가하면 자동으로 분석해서 추천값 적용")
        self.auto_on_add.setChecked(self.main.cfg.get("photo_auto", True))
        fl.addWidget(self.auto_on_add)
        lay.addWidget(files)

        ana = QGroupBox("분석 · 추천")
        al = QVBoxLayout(ana)
        self.analysis = QLabel("사진을 추가하면 압축 상태 · 크기 · 얼굴을 분석해 알맞은 설정을 추천합니다.")
        self.analysis.setWordWrap(True)
        self.analysis.setTextFormat(Qt.TextFormat.RichText)
        self.analysis.setObjectName("muted")
        al.addWidget(self.analysis)
        # 분석 결과와 [자동 설정] 을 한곳에
        self.btn_auto = button("자동 설정", "sparkles",
                               tip="적용 범위의 사진들을 분석해 사진마다 추천값으로 설정합니다 (스타일 취향은 유지)")
        self.btn_auto.clicked.connect(self._auto)
        al.addWidget(self.btn_auto, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(ana)

        st = QGroupBox("설정")
        sl = QVBoxLayout(st)
        scope = QHBoxLayout()
        self.scope = Segmented([("일괄 · 모든 사진", "all"), ("개별 · 선택한 사진만", "one")],
                               "일괄: 바꾼 항목이 모든 사진에 적용 (바꾸지 않은 항목은 사진별 값 유지)\n"
                               "개별: 목록에서 고른 사진에만 적용 (Ctrl·Shift 로 여러 장 선택)")
        _set_combo(self.scope, "one" if self.main.cfg.get("photo_mode") == "one" else "all")
        self.scope.currentIndexChanged.connect(lambda _: self._update_scope())
        scope.addWidget(QLabel("적용 범위"))
        scope.addWidget(self.scope, 1)
        sl.addLayout(scope)
        self.scope_info = QLabel()
        self.scope_info.setStyleSheet(f"color:{ui.ACCENT};font-size:8.5pt")
        self.scope_info.setWordWrap(True)
        sl.addWidget(self.scope_info)
        tabs = QTabWidget()
        tabs.setIconSize(QSize(15, 15))
        tabs.addTab(self._tab_basic(), icon("sliders", ui.MUTED), "기본")
        tabs.addTab(self._tab_restore(), icon("sparkles", ui.MUTED), "복원")
        tabs.addTab(self._tab_style(), icon("palette", ui.MUTED), "스타일")
        tabs.addTab(self._tab_save(), icon("export", ui.MUTED), "저장")
        sl.addWidget(tabs)
        row = QHBoxLayout()
        b_reset = button("기본값", "reset", tip="적용 범위의 사진들을 기본값으로 되돌립니다")
        b_reset.clicked.connect(self._reset)
        self.btn_copy = button("모든 사진에 복사", "copy",
                               tip="선택한 사진의 설정 전체를 목록의 모든 사진에 그대로 복사합니다")
        self.btn_copy.clicked.connect(self._copy_to_all)
        for b in (b_reset, self.btn_copy):
            row.addWidget(b)
        sl.addLayout(row)
        lay.addWidget(st, 1)
        return w

    def _form(self):
        page = QWidget()
        f = form_layout(page)
        f.setContentsMargins(4, 8, 4, 4)
        return page, f

    def _tab_basic(self):
        page, f = self._form()
        self.quality = _combo(PHOTO_QUALITY_ITEMS)
        self.size = _combo(PHOTO_SIZE_ITEMS, "결과 사진 크기")
        self.size_custom = QSpinBox()
        self.size_custom.setRange(256, 16384)
        self.size_custom.setSingleStep(64)
        self.size_custom.setSuffix(" px")
        size_row = QHBoxLayout()
        size_row.addWidget(self.size, 1)
        size_row.addWidget(self.size_custom)
        f.addRow("품질", self.quality)
        f.addRow("출력 크기", size_row)
        f.addRow(note("사진은 원본 해상도 그대로(처리 해상도 100%) AI 에 넣어 디테일을 최대한 살립니다."))
        for c in (self.quality, self.size):
            c.currentIndexChanged.connect(self._changed)
        self.size_custom.valueChanged.connect(self._changed)
        return page

    def _tab_restore(self):
        page, f = self._form()
        self.denoise = Slider("압축(JPG) 깨짐·노이즈를 지우는 강도 (표준 모델 전용)")
        self.texture = Slider("AI 결과에 원본을 섞어 인위적인 느낌을 줄입니다")
        self.face_enhance = Toggle()
        self.face_enhance.setEnabled(face_mod.available())
        self.face_strength = Slider("높을수록 선명하지만 얼굴이 조금 달라 보일 수 있습니다")
        restore = FeatureCard("sparkles", "AI 화질 복원", "압축 깨짐·노이즈를 지우고 디테일을 살립니다")
        restore.form.addRow("노이즈 제거", self.denoise)
        restore.form.addRow("원본 질감 유지", self.texture)
        self.face_card = FeatureCard("face", "얼굴 복원",
                                     "눈·피부·치아를 선명하게. 사진은 깜빡임 걱정이 없어 조금 더 강하게 써도 됩니다",
                                     self.face_enhance)
        self.face_card.form.addRow("강도", self.face_strength)
        f.addRow(restore)
        f.addRow(self.face_card)
        self.face_enhance.toggled.connect(self._changed)
        for s in (self.denoise, self.texture, self.face_strength):
            s.changed.connect(self._changed)
        return page

    def _tab_style(self):
        page, f = self._form()
        # 스타일 프리셋은 그 프리셋이 바꾸는 값들과 같은 탭에 둔다
        self.style = _combo([(k, k) for k in pipeline.STYLE_PRESETS] + [(CUSTOM, CUSTOM)],
                            "아래 값들을 한 번에 바꿉니다. 직접 바꾸면 '사용자 지정' 이 됩니다")
        self.style.activated.connect(self._on_style)
        self.soft_skin, self.glow, self.brighten = Slider(), Slider(), Slider()
        self.sharpen, self.grain = Slider(), Slider()
        f.addRow("스타일 프리셋", self.style)
        f.addRow("피부 보정", self.soft_skin)
        f.addRow("빛 번짐", self.glow)
        f.addRow("화사함", self.brighten)
        f.addRow("선명도", self.sharpen)
        f.addRow("필름 그레인", self.grain)
        for s in (self.soft_skin, self.glow, self.brighten, self.sharpen, self.grain):
            s.changed.connect(self._changed)
        return page

    def _tab_save(self):
        page, f = self._form()
        self.fmt = _combo(PHOTO_FORMAT_ITEMS)
        self.jpg_quality = QSpinBox()
        self.jpg_quality.setRange(50, 100)
        self.jpg_quality.setToolTip("JPG · WEBP 화질 (95 추천)")
        self.keep_exif = Toggle("촬영 정보(EXIF: 날짜·카메라 등) 유지")
        self.out_dir = QLineEdit(self.main.cfg.get("photo_out_dir", ""))
        self.out_dir.setPlaceholderText("비워 두면 원본과 같은 폴더 (모든 사진 공통)")
        b = button("찾기", "folder")
        b.clicked.connect(self._pick_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(b)
        self.open_after = Toggle("완료 후 결과 폴더 열기")
        self.open_after.setChecked(self.main.cfg.get("photo_open_after", True))
        f.addRow("파일 형식", self.fmt)
        f.addRow("화질 (JPG · WEBP)", self.jpg_quality)
        f.addRow(self.keep_exif)
        f.addRow("저장 위치", out_row)
        f.addRow(self.open_after)
        f.addRow(note("파일 이름: 원본이름_개선_2x.jpg 처럼 저장되며 원본은 바뀌지 않습니다."))
        self.fmt.currentIndexChanged.connect(self._changed)
        self.jpg_quality.valueChanged.connect(self._changed)
        self.keep_exif.toggled.connect(self._changed)
        return page

    # ── 오른쪽 ──
    def _build_right(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 4, 8)
        self.view = CompareView()
        self.view.set_message("사진을 추가하고 [미리보기] 를 눌러 보세요")
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.size_info = QLabel("")
        self.size_info.setObjectName("muted")
        row.addWidget(self.size_info, 1)
        self.btn_preview = button("미리보기 (선택한 사진)", "eye")
        self.btn_preview.clicked.connect(self._preview)
        row.addWidget(self.btn_preview)
        lay.addLayout(row)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        bottom = QHBoxLayout()
        self.status = QLabel("준비됨")
        self.status.setObjectName("muted")
        bottom.addWidget(self.status, 1)
        self.btn_open = button("결과 폴더", "folder")
        self.btn_open.clicked.connect(self._open_out)
        self.btn_start = button("모두 변환", "play", "primary")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = button("중지", "stop", "danger")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        bottom.addWidget(self.btn_open)
        bottom.addWidget(self.btn_stop)
        bottom.addWidget(self.btn_start)
        lay.addLayout(bottom)
        return w

    # ── 설정 ↔ 화면 ──
    def _read_settings(self):
        s = photo_mod.PhotoSettings()
        s.quality = self.quality.currentData()
        s.size = self.size.currentData()
        s.size_custom = self.size_custom.value()
        for k in ("denoise", "texture", "face_strength", "soft_skin", "glow", "brighten", "sharpen", "grain"):
            setattr(s, k, round(getattr(self, k).value(), 2))
        s.face_enhance = self.face_enhance.isChecked()
        s.fmt = self.fmt.currentData()
        s.jpg_quality = self.jpg_quality.value()
        s.keep_exif = self.keep_exif.isChecked()
        return s

    def _show_settings(self, s):
        self._loading = True
        try:
            _set_combo(self.quality, s.quality)
            _set_combo(self.size, s.size)
            self.size_custom.setValue(s.size_custom)
            for k in ("denoise", "texture", "face_strength", "soft_skin", "glow", "brighten", "sharpen", "grain"):
                getattr(self, k).setValue(getattr(s, k))
            self.face_enhance.setChecked(s.face_enhance and face_mod.available())
            _set_combo(self.fmt, s.fmt)
            self.jpg_quality.setValue(s.jpg_quality)
            self.keep_exif.setChecked(s.keep_exif)
        finally:
            self._loading = False
        self._shown = self._read_settings().to_dict()
        self._sync_widgets()

    def _sync_widgets(self):
        self.size_custom.setVisible(self.size.currentData() == "custom")
        self.denoise.setEnabled(self.quality.currentData() == "general_v3")
        self.face_strength.setEnabled(self.face_enhance.isChecked())
        self.face_card.sync()
        self.jpg_quality.setEnabled(self.fmt.currentData() in ("same", ".jpg", ".webp"))
        match = next((n for n, d in pipeline.STYLE_PRESETS.items()
                      if all(abs(getattr(self, k).value() - v) < 0.02 for k, v in d.items() if hasattr(self, k))),
                     CUSTOM)
        self.style.blockSignals(True)
        _set_combo(self.style, match)
        self.style.blockSignals(False)

    def _on_style(self, _):
        name = self.style.currentData()
        if name in pipeline.STYLE_PRESETS:
            self._loading = True
            for k, v in pipeline.STYLE_PRESETS[name].items():
                if hasattr(self, k):
                    getattr(self, k).setValue(v)
            self._loading = False
        self._changed()

    def _targets(self):
        """지금 설정을 바꾸면 적용될 사진들."""
        if self.scope.currentData() == "all":
            return [self.files.item(i) for i in range(self.files.count())]
        return self.files.selectedItems()

    def _update_scope(self):
        n, sel = self.files.count(), len(self.files.selectedItems())
        if self.scope.currentData() == "all":
            self.scope_info.setText(f"바꾸는 항목이 모든 사진 ({n}장) 에 적용됩니다 · 바꾸지 않은 항목은 사진별 값 유지")
        else:
            self.scope_info.setText(f"선택한 사진 ({sel}장) 에만 적용됩니다 · Ctrl·Shift 로 여러 장 선택")
        self.btn_copy.setEnabled(n > 1)

    def _changed(self, *_):
        if self._loading:
            return
        new = self._read_settings().to_dict()
        diff = {k: v for k, v in new.items() if self._shown is None or self._shown.get(k) != v}
        self._shown = new
        self._sync_widgets()
        if diff:
            for it in self._targets():
                d = dict(it.data(ROLE_SET))
                d.update(diff)
                it.setData(ROLE_SET, d)
                self._refresh_item(it)
        self._update_info()

    def _state_of(self, it):
        d, reco = it.data(ROLE_SET), it.data(ROLE_RECO)
        if reco:
            return "자동" if all(d.get(k) == v for k, v in reco.items()) else "수정됨"
        return "기본" if d == photo_mod.PhotoSettings().to_dict() else "수정됨"

    def _refresh_item(self, it):
        wdg = self.files.itemWidget(it)
        if wdg is None:
            return
        w, h = it.data(ROLE_INFO)
        s = photo_mod.PhotoSettings.from_dict(it.data(ROLE_SET))
        ow, oh = s.output_size(w, h)
        q = "고품질" if s.quality == "x4plus" else "표준"
        face = " · 얼굴복원" if s.face_enhance else ""
        wdg.detail.setText(f"{w}x{h} → {ow}x{oh} · {q}{face}")
        wdg.set_state(self._state_of(it))

    # ── 자동 · 기본값 · 복사 ──
    def _auto(self, items=None):
        items = items if items is not None else self._targets()
        if not items:
            self.status.setText("자동 설정할 사진을 선택하세요")
            return
        if self.thread is not None:
            self.status.setText("다른 작업 중입니다")
            return
        jobs = [(it.data(ROLE_PATH), photo_mod.PhotoSettings.from_dict(it.data(ROLE_SET))) for it in items]
        for it in items:
            self.files.itemWidget(it).set_state("분석 중")
        results = {}

        def job(w):
            for n, (p, cur) in enumerate(jobs, 1):
                results[p] = photo_mod.recommend(load_image_rgb(p), cur)
                w.progress.emit(p, n, len(jobs))
            return f"자동 설정 완료 ({len(jobs)}장)"

        def done(_):
            for p, (reco, why) in results.items():
                it = self._find(p)
                if it is None:
                    continue
                d = dict(it.data(ROLE_SET))
                d.update(reco)
                it.setData(ROLE_SET, d)
                it.setData(ROLE_RECO, reco)
                it.setData(ROLE_WHY, why)
                self._refresh_item(it)
            self._on_row_changed(self.files.currentRow())
        self._run(job, done)

    def _reset(self):
        default = photo_mod.PhotoSettings().to_dict()
        for it in self._targets():
            it.setData(ROLE_SET, dict(default))
            self._refresh_item(it)
        self._on_row_changed(self.files.currentRow())
        self.status.setText("기본값으로 되돌렸습니다")

    def _copy_to_all(self):
        cur = self.files.currentItem()
        if cur is None:
            return
        d = cur.data(ROLE_SET)
        for i in range(self.files.count()):
            self.files.item(i).setData(ROLE_SET, dict(d))
            self._refresh_item(self.files.item(i))
        self.status.setText(f"'{Path(cur.data(ROLE_PATH)).name}' 의 설정을 모든 사진 ({self.files.count()}장) 에 복사했습니다")

    # ── 백그라운드 작업 (사진 탭 전용) ──
    def _run(self, fn, on_done=None):
        self._on_done = on_done
        self.stop_event = threading.Event()
        self.thread = QThread()
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # 메인 창의 메서드(슬롯)에 연결해야 화면 스레드에서 실행된다
        self.worker.progress.connect(self._on_progress)
        self.worker.preview.connect(self.view.set_images)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._busy(True)
        self.thread.start()

    def _cleanup(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        self._busy(False)

    @Slot(object, int, int)
    def _on_progress(self, info, n, total):
        if isinstance(info, tuple):  # 처리 진행: (사진 경로, 0~100, 설명)
            path, pct, text = info
            it = self._find(path)
            if it is not None:
                self.files.itemWidget(it).set_progress(pct, "완료" if pct >= 100 else f"{int(pct)}%")
            prefix = f"[{n}/{total}] " if total > 1 else ""
            self.status.setText(f"{prefix}{Path(path).name} — {text}")
            self.bar.setMaximum(max(1, total) * 100)
            self.bar.setValue(int(max(0, n - 1) * 100 + pct) if total > 1 else int(pct))
        else:  # 자동 분석 진행
            self.status.setText(f"사진 분석 중… {n}/{total}")
            self.bar.setMaximum(total)
            self.bar.setValue(n)

    @Slot(str)
    def _on_finished(self, msg):
        on_done, self._on_done = self._on_done, None
        self._cleanup()
        self.status.setText(msg)
        if on_done:
            on_done(msg)

    @Slot(str)
    def _on_failed(self, msg):
        self._on_done = None
        self._cleanup()
        self.status.setText("오류")
        QMessageBox.critical(self, "오류", msg[-3000:])

    # ── 파일 ──
    def _find(self, path):
        for i in range(self.files.count()):
            if self.files.item(i).data(ROLE_PATH) == path:
                return self.files.item(i)
        return None

    def add_files(self, paths):
        existing = {self.files.item(i).data(ROLE_PATH) for i in range(self.files.count())}
        base = self._read_settings().to_dict() if self.scope.currentData() == "all" and self.files.count() \
            else photo_mod.PhotoSettings().to_dict()
        added = []
        for p in paths:
            if p in existing:
                continue
            try:
                img = load_image_rgb(p)
            except Exception as e:
                QMessageBox.warning(self, "열 수 없음", f"{p}\n\n{e}")
                continue
            h, w = img.shape[:2]
            k = 64 / max(w, h)
            thumb = _qimage(cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA))
            it = QListWidgetItem()
            it.setData(ROLE_PATH, p)
            it.setData(ROLE_INFO, (w, h))
            it.setData(ROLE_SET, dict(base))
            it.setToolTip(p)
            wdg = PhotoItem(thumb, Path(p).name)
            from PySide6.QtCore import QSize
            it.setSizeHint(QSize(0, 84))  # 썸네일 64px + 진행 막대가 잘리지 않는 높이
            self.files.addItem(it)
            self.files.setItemWidget(it, wdg)
            self._refresh_item(it)
            added.append(it)
        if self.files.currentRow() < 0 and self.files.count():
            self.files.setCurrentRow(0)
        self._update_scope()
        if added and self.auto_on_add.isChecked():
            self._auto(added)

    def _add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "사진 선택", self.main.cfg.get("last_photo_dir", ""),
                                                "사진 (" + " ".join("*" + e for e in IMAGE_EXT) + ")")
        if paths:
            self.main.cfg["last_photo_dir"] = str(Path(paths[0]).parent)
        self.add_files(paths)

    def _remove_selected(self):
        for it in self.files.selectedItems():
            self.files.takeItem(self.files.row(it))
        self._on_row_changed(self.files.currentRow())

    def _clear(self):
        self.files.clear()
        self.view.set_images(None, None)
        self._on_row_changed(-1)

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _on_row_changed(self, row):
        it = self.files.item(row) if row >= 0 else None
        if it is not None:
            self._show_settings(photo_mod.PhotoSettings.from_dict(it.data(ROLE_SET)))
            why = it.data(ROLE_WHY)
            self.analysis.setText("<br>".join("• " + x for x in why) if why else
                                  "아직 분석하지 않았습니다. [자동 설정] 을 누르면 분석합니다.")
            try:
                self.view.set_images(None, load_image_rgb(it.data(ROLE_PATH)))  # 처리 전에는 원본만
            except Exception:
                pass
        else:
            self.analysis.setText("사진을 추가하면 압축 상태 · 크기 · 얼굴을 분석해 알맞은 설정을 추천합니다.")
        self._update_scope()
        self._update_info()

    def _update_info(self):
        it = self.files.currentItem()
        if it is None:
            self.size_info.setText("")
            return
        w, h = it.data(ROLE_INFO)
        ow, oh = photo_mod.PhotoSettings.from_dict(it.data(ROLE_SET)).output_size(w, h)
        self.size_info.setText(f"{Path(it.data(ROLE_PATH)).name} · 원본 {w}x{h} → 결과 <b>{ow}x{oh}</b> ({ow / w:.1f}배)")

    # ── 처리 ──
    def _cache_key(self, it):
        return it.data(ROLE_PATH), json.dumps(it.data(ROLE_SET), sort_keys=True)

    def _remember(self, key, result):
        """최근 미리보기 결과를 기억 → 설정을 안 바꾸고 변환하면 다시 처리하지 않음 (메모리 위해 2장까지)."""
        self._cache[key] = result
        while len(self._cache) > 2:
            self._cache.pop(next(iter(self._cache)))

    def _busy(self, on):
        for b in (self.btn_preview, self.btn_start, self.btn_auto):
            b.setEnabled(not on)
        self.btn_stop.setEnabled(on)

    def _preview(self):
        it = self.files.currentItem()
        if it is None:
            QMessageBox.information(self, "알림", "먼저 사진을 추가하고 선택하세요.")
            return
        if self.thread is not None:
            return
        path, key = it.data(ROLE_PATH), self._cache_key(it)
        s = photo_mod.PhotoSettings.from_dict(it.data(ROLE_SET))
        dev = self.main.device.currentData() if hasattr(self.main, "device") else "auto"
        shown = {}

        def job(w):
            t0 = time.time()
            img = load_image_rgb(path)
            result = self._cache.get(key)
            if result is None:
                result = photo_mod.enhance(img, s, dev, lambda p, t: w.progress.emit((path, p, t), 0, 1),
                                           self.stop_event)
            shown["result"] = result
            before = cv2.resize(img, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
            w.preview.emit(before, result)
            return f"미리보기 완료 ({time.time() - t0:.1f}초) — {result.shape[1]}x{result.shape[0]}"

        def done(_):
            self._remember(key, shown["result"])
            self.files.itemWidget(it).set_progress(0, "미리보기 완료")
        self.view.set_message("처리 중…")
        self._run(job, done)

    def _start(self):
        if not self.files.count():
            QMessageBox.information(self, "알림", "먼저 사진을 추가하세요.")
            return
        if self.thread is not None:
            return
        jobs = []
        for i in range(self.files.count()):
            it = self.files.item(i)
            jobs.append((it.data(ROLE_PATH), photo_mod.PhotoSettings.from_dict(it.data(ROLE_SET)),
                         self._cache.get(self._cache_key(it))))
            self.files.itemWidget(it).set_progress(0, "대기")
        out_dir = self.out_dir.text()
        dev = self.main.device.currentData() if hasattr(self.main, "device") else "auto"
        saved, failed = [], []

        def job(w):
            for n, (path, s, cached) in enumerate(jobs, 1):
                prog = lambda p, t, path=path: w.progress.emit((path, p, t), n, len(jobs))
                try:
                    result = cached if cached is not None else \
                        photo_mod.enhance(load_image_rgb(path), s, dev, prog, self.stop_event)
                    prog(99, "저장 중")
                    saved.append(photo_mod.save(result, path, s, out_dir))
                    prog(100, "완료")
                except photo_mod.Stopped:
                    raise pipeline.Stopped
                except Exception as e:  # 한 장이 실패해도 나머지는 계속
                    log.error("사진 처리 실패 %s\n%s", path, traceback.format_exc())
                    failed.append((path, str(e)))
                    prog(0, "실패")
            msg = f"완료: {len(saved)}장 저장"
            return msg + (f", 실패 {len(failed)}장" if failed else "")

        def done(_):
            self._last_out = str(saved[-1].parent) if saved else None
            if failed:
                QMessageBox.warning(self, "일부 실패", "\n\n".join(f"{Path(p).name}\n{e}" for p, e in failed)[-3000:])
            if saved and self.open_after.isChecked():
                self._open_out()
        self._run(job, done)

    def _stop(self):
        self.stop_event.set()
        self.status.setText("중지하는 중…")

    def _open_out(self):
        d = self.out_dir.text().strip() or getattr(self, "_last_out", None)
        if not d and self.files.currentItem() is not None:
            d = str(Path(self.files.currentItem().data(ROLE_PATH)).parent)
        if d:
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))


# ── 메인 창 ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle(f"AI 영상·사진 업스케일러 {__version__}")
        self.resize(1400, 920)
        self.setAcceptDrops(True)
        self.thread = self.worker = None
        self._on_done = None
        self.stop_event = threading.Event()
        self._loading = False
        self._last_preset = {}  # 프리셋 콤보 → 마지막으로 고른 프리셋 이름
        self._before_filter = None  # '필터만' 으로 바꾸기 전의 값 (AI 로 돌아오면 되돌림)
        self.cfg = self._load_cfg()

        root = QSplitter()
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left.setWidget(self._build_left())
        left.setMinimumWidth(460)
        root.addWidget(left)
        root.addWidget(self._build_right())
        root.setStretchFactor(1, 1)
        root.setSizes([490, 910])

        # 맨 위: 로고 · 이름 + [영상] [사진] 탭
        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("mainTabs")
        self.main_tabs.setIconSize(QSize(17, 17))
        self.main_tabs.addTab(root, icon("video", "#ffffff"), "영상")
        self.photo = PhotoPanel(self)
        self.main_tabs.addTab(self.photo, icon("image", "#ffffff"), "사진")
        brand = QWidget()
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(14, 0, 18, 0)
        bl.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(ui.app_pixmap(28))
        name = QLabel("AI 업스케일러")
        name.setStyleSheet("font-weight:700;font-size:11pt")
        bl.addWidget(logo)
        bl.addWidget(name)
        self.main_tabs.setCornerWidget(brand, Qt.Corner.TopLeftCorner)
        self.setCentralWidget(self.main_tabs)

        # 저작권 표시 + 정보
        about = button("정보", "info", "ghost")
        about.clicked.connect(self._about)
        mark = QLabel(f"v{__version__}  ·  © 2026 최승훈. All rights reserved.")
        mark.setStyleSheet(f"color:{ui.FAINT};font-size:8.5pt;padding-right:6px")
        self.statusBar().addPermanentWidget(mark)
        self.statusBar().addPermanentWidget(about)

        self._apply_settings(pipeline.Settings.from_dict(self.cfg.get("settings", {})))
        self.out_dir.setText(self.cfg.get("out_dir", ""))
        _set_combo(self.container, self.cfg.get("container", ".mp4"))
        self.auto_analyze.setChecked(self.cfg.get("auto_analyze", True))
        self.open_after.setChecked(self.cfg.get("open_after", True))
        self._sync_preset_labels()
        self._show_analysis(None)
        self._update_info()

    # ── 화면 구성 ──
    def _build_left(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 6, 8)

        files = QGroupBox("영상 파일")
        fl = QVBoxLayout(files)
        self.files = DropList("영상이나 폴더를 여기에 끌어다 놓으세요")
        self.files.setMinimumHeight(96)
        self.files.setMaximumHeight(140)
        self.files.currentRowChanged.connect(self._on_row_changed)
        fl.addWidget(self.files)
        row = QHBoxLayout()
        for text, ic, fn in (("추가", "plus", self._add_dialog), ("삭제", "minus", self._remove_selected),
                             ("모두 지우기", "trash", self._clear_files)):
            b = button(text, ic)
            b.clicked.connect(fn)
            row.addWidget(b)
        fl.addLayout(row)
        self.auto_analyze = Toggle("영상을 추가하면 자동으로 분석해서 추천값 적용")
        self.auto_analyze.toggled.connect(lambda _: self._save_cfg())
        fl.addWidget(self.auto_analyze)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("muted")
        fl.addWidget(self.info)
        lay.addWidget(files)

        # 분석 결과와 [추천값 적용] 을 한곳에
        ana = QGroupBox("분석 · 추천")
        al = QVBoxLayout(ana)
        self.analysis = QLabel()
        self.analysis.setWordWrap(True)
        self.analysis.setTextFormat(Qt.TextFormat.RichText)
        self.analysis.setObjectName("muted")
        al.addWidget(self.analysis)
        self.btn_reco = button("추천값 적용", "sparkles", tip="선택한 영상을 분석해 추천 설정으로 바꿉니다 (스타일은 유지)")
        self.btn_reco.clicked.connect(self._apply_recommendation)
        al.addWidget(self.btn_reco, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(ana)

        st = QGroupBox("설정 · 선택한 파일에 적용")
        sl = QVBoxLayout(st)
        self.tabs = QTabWidget()
        self.tabs.setIconSize(QSize(15, 15))
        self.tabs.addTab(self._tab_basic(), icon("sliders", ui.MUTED), "기본")
        self.tabs.addTab(self._tab_restore(), icon("sparkles", ui.MUTED), "복원")
        self.tabs.addTab(self._tab_interp(), icon("frames", ui.MUTED), "보간")
        self.tabs.addTab(self._tab_style(), icon("palette", ui.MUTED), "스타일")
        self.tabs.addTab(self._tab_output(), icon("export", ui.MUTED), "출력")
        sl.addWidget(self.tabs)
        row = QHBoxLayout()
        self.btn_reset = button("기본값", "reset", tip="모든 설정을 처음 기본값으로 되돌립니다")
        self.btn_reset.clicked.connect(self._reset_defaults)
        self.btn_all = button("모든 파일에 적용", "copy", tip="지금 설정을 목록의 모든 영상에 똑같이 적용합니다")
        self.btn_all.clicked.connect(self._apply_to_all)
        for b in (self.btn_reset, self.btn_all):
            row.addWidget(b)
        sl.addLayout(row)
        lay.addWidget(st, 1)
        return w

    def _form_tab(self):
        page = QWidget()
        form = form_layout(page)
        form.setContentsMargins(4, 8, 4, 4)
        return page, form

    def _tab_basic(self):
        page, f = self._form_tab()
        self.res = _combo(RES_ITEMS, "출력 해상도 (짧은 변 기준, 유튜브 화질과 같은 방식)")
        self.res_custom = QSpinBox()
        self.res_custom.setRange(144, 8640)
        self.res_custom.setSingleStep(2)
        self.res_custom.setSuffix(" p")
        self.fps = _combo(FPS_ITEMS, "출력 프레임레이트. 원본보다 높으면 AI 로 중간 프레임을 만듭니다")
        self.fps_custom = QDoubleSpinBox()
        self.fps_custom.setRange(1, 240)
        self.fps_custom.setDecimals(3)
        self.fps_custom.setSuffix(" fps")
        # 연산 장치는 업스케일·얼굴 복원용과 보간용을 한 카드에 모은다
        devices = [(name, dev) for dev, name in backends.list_devices()]  # 콤보는 (표시 이름, 값)
        self.device = _combo([("자동 (가장 빠른 장치)", "auto")] + devices)
        self.device_rife = _combo([("자동", "auto")] + devices,
                                  "자동: Intel 내장 GPU 로 업스케일할 때는 보간을 CPU 에 맡겨 동시에 처리합니다")
        res_row, fps_row = QHBoxLayout(), QHBoxLayout()
        res_row.addWidget(self.res, 1)
        res_row.addWidget(self.res_custom)
        fps_row.addWidget(self.fps, 1)
        fps_row.addWidget(self.fps_custom)
        f.addRow("출력 해상도", res_row)
        f.addRow("프레임레이트", fps_row)

        self.my_presets = QComboBox()
        self._reload_presets()
        pr = QHBoxLayout()
        pr.addWidget(self.my_presets, 1)
        for tip, ic, fn in (("불러오기", "download", self._load_preset), ("지금 설정을 저장", "save", self._save_preset),
                            ("삭제", "trash", self._delete_preset)):
            b = button("", ic, tip=tip)
            b.setFixedWidth(36)
            b.clicked.connect(fn)
            pr.addWidget(b)
        f.addRow("내 프리셋", pr)
        self.dev_card = FeatureCard("bolt", "연산 장치", "보통은 '자동' 그대로 두세요")
        self.dev_card.form.addRow("업스케일 · 얼굴", self.device)
        self.dev_card.form.addRow("보간", self.device_rife)
        f.addRow(self.dev_card)

        for c in (self.res, self.fps, self.device, self.device_rife):
            c.currentIndexChanged.connect(self._changed)
        self.res_custom.valueChanged.connect(self._changed)
        self.fps_custom.valueChanged.connect(self._changed)
        return page

    def _tab_restore(self):
        page, f = self._form_tab()
        # 처리 방식은 여기 한 곳에서만 고른다 (AI 를 쓸지 · 어떻게 쓸지)
        self.mode = Segmented([("AI 복원", "ai"), ("보간만", "interp"), ("필터만", "filter")])
        self.mode_desc = note("")
        self.mode_desc.setStyleSheet(f"color:{ui.ACCENT}")
        f.addRow("처리 방식", self.mode)
        f.addRow(self.mode_desc)
        self.mode.currentIndexChanged.connect(self._on_engine)
        # AI 복원의 화질·속도 프리셋은 그 값들(아래 카드)과 같은 탭에 둔다
        self.quality = _combo([(k, k) for k in pipeline.QUALITY_PRESETS] + [(CUSTOM, CUSTOM)],
                              "아래 AI 설정(처리 해상도·모델·얼굴 수 등)을 한 번에 바꿉니다. 권장 출력도 함께 들어갑니다")
        self.quality.activated.connect(lambda _: self._on_preset(self.quality, pipeline.QUALITY_PRESETS))
        f.addRow("품질/속도", self.quality)
        self.quality_label = f.labelForField(self.quality)
        self.sr_model = _combo([(v["label"], k) for k, v in esrgan.available_models().items()],
                               "고품질 모델은 디테일(머리카락·옷감 등)이 훨씬 좋지만 20배 이상 느립니다")
        self.proc_scale = _combo(PROC_ITEMS, "AI 가 처리할 해상도 (원본 대비)")
        self.sr_int8 = Toggle("약 1.45배 빠름 · 미세 노이즈 약간")
        self.sr_int8.setToolTip("General v3 모델 + Intel 장치(OpenVINO)에서만 동작합니다. "
                                "압축 노이즈 제거는 45/60/75/90 중 가장 가까운 단계로 적용됩니다 (45 미만은 45)")
        self.sr_int8.setEnabled(esrgan.int8_available())
        self.denoise = Slider("블록·모자이크 같은 압축 깨짐을 지우는 강도. 너무 높으면 뭉개짐 (General v3 전용)")
        self.texture = Slider("AI 결과에 원본을 섞어 인위적인 느낌을 줄입니다")
        self.face_enhance = Toggle()
        self.face_enhance.setEnabled(face_mod.available())
        self.face_strength = Slider("높을수록 선명하지만 얼굴이 조금 달라 보일 수 있습니다 (50~70 추천)")
        self.face_min = QSpinBox()
        self.face_min.setRange(16, 400)
        self.face_min.setSuffix(" px")
        self.face_min.setToolTip("이보다 작은 얼굴(업스케일 후 기준)은 복원하지 않습니다. 너무 작은 얼굴은 어색해지기 쉽습니다")
        self.face_max = QSpinBox()
        self.face_max.setRange(0, 20)
        self.face_max.setSpecialValueText("제한 없음")
        self.face_max.setSuffix(" 개")
        self.face_max.setToolTip("한 화면에서 큰 얼굴부터 이 개수만 복원합니다. 단체 샷에서 처리 시간을 크게 줄입니다")
        self.deinterlace = Segmented([("자동", "auto"), ("켜기", "on"), ("끄기", "off")],
                                     "TV 방송 녹화본처럼 줄무늬(인터레이스)가 있는 영상용")
        # AI 복원 모드에서만 보이는 카드들 (얼굴 복원은 토글로 켜고 끔)
        self.sr_card = FeatureCard("sparkles", "AI 화질 복원 · 업스케일",
                                   "압축 깨짐을 지우고 디테일을 살려 해상도를 키웁니다")
        self.sr_card.form.addRow("업스케일 모델", self.sr_model)
        self.sr_card.form.addRow("처리 해상도", self.proc_scale)
        self.sr_card.form.addRow("AI 경량화 (INT8)", self.sr_int8)
        self.sr_card.form.addRow("원본 질감 유지", self.texture)
        self.sr_card.form.addRow(note("처리 해상도가 낮을수록 빠르고 압축 깨짐도 덜 보입니다. AI 는 처리 해상도의 "
                                      "모델 배율(4배 또는 x2plus 는 2배)까지 만들 수 있으며, 그 이상은 단순 확대됩니다."))
        self.face_card = FeatureCard("face", "얼굴 복원",
                                     "눈·피부·치아를 선명하게 — 음악방송에서 효과가 가장 큽니다", self.face_enhance)
        self.face_card.form.addRow("강도", self.face_strength)
        self.face_card.form.addRow("최소 얼굴 크기", self.face_min)
        self.face_card.form.addRow("최대 얼굴 수", self.face_max)
        self.common_card = FeatureCard("layers", "공통", "")
        self.common_card.form.addRow("압축 노이즈 제거", self.denoise)
        self.common_card.form.addRow("인터레이스 제거", self.deinterlace)
        f.addRow(self.sr_card)
        f.addRow(self.face_card)
        f.addRow(self.common_card)
        self.sr_int8.toggled.connect(self._changed)
        self.face_enhance.toggled.connect(self._changed)
        self.face_min.valueChanged.connect(self._changed)
        self.face_max.valueChanged.connect(self._changed)
        for c in (self.sr_model, self.proc_scale, self.deinterlace):
            c.currentIndexChanged.connect(self._changed)
        for s in (self.denoise, self.texture, self.face_strength):
            s.changed.connect(self._changed)
        return page

    def _tab_interp(self):
        page, f = self._form_tab()
        self.rife_model = _combo([(v, k) for k, v in pipeline.RIFE_MODELS.items()])
        self.scene = Slider("높을수록 카메라 전환을 민감하게 감지 (0 = 끄기). "
                            "조명 깜빡임에 보간이 끊기면 낮추세요")
        self.interp_res = _combo([("자동 (작은 얼굴 구간만 원본 해상도)", "auto"),
                                  ("빠른 자동 (작은 얼굴 주변만 원본 해상도)", "fast"),
                                  ("항상 원본 해상도 (가장 정확·느림)", "full"),
                                  ("항상 처리 해상도 (빠름·작은 얼굴이 깜빡일 수 있음)", "proc")],
                                 "얼굴 복원을 켰을 때 중간 프레임을 만드는 해상도")
        self.dedup = Toggle("중복 프레임 보정 (줌·이동 중 끊김 방지)")
        self.dedup.setToolTip("방송 녹화·재인코딩 영상에 섞인 같은 프레임을 건너뛰고 앞뒤 프레임 사이를 보간합니다")
        # 보간하지 않는 설정(원본 프레임레이트 · 필터만)이면 이유를 보여 주고 설정을 막는다
        self.interp_off = note("")
        self.interp_off.setStyleSheet(f"color:{ui.ACCENT}")
        f.addRow(self.interp_off)
        f.addRow("보간 모델", self.rife_model)
        f.addRow("장면 전환 감지", self.scene)
        f.addRow(self.dedup)
        f.addRow("보간 해상도", self.interp_res)
        f.addRow(note("프레임레이트는 '기본' 탭에서 고릅니다. 장면 전환 감지는 카메라가 바뀌는 순간 "
                      "두 장면이 섞인 이상한 중간 프레임이 생기지 않게 합니다. 줌·이동·조명 깜빡임은 "
                      "컷으로 보지 않습니다."))
        self.dedup.toggled.connect(self._changed)
        self.interp_res.currentIndexChanged.connect(self._changed)
        self.rife_model.currentIndexChanged.connect(self._changed)
        self.scene.changed.connect(self._changed)
        return page

    def _tab_style(self):
        page, f = self._form_tab()
        # 스타일 프리셋은 그 프리셋이 바꾸는 값들과 같은 탭에 둔다
        self.style = _combo([(k, k) for k in pipeline.STYLE_PRESETS] + [(CUSTOM, CUSTOM)],
                            "아래 값들을 한 번에 바꿉니다. 직접 바꾸면 '사용자 지정' 이 됩니다")
        self.soft_skin = Slider("피부를 매끈하게 (뽀샤시)")
        self.glow = Slider("밝은 부분이 은은하게 퍼지는 빛 번짐 (뽀샤시)")
        self.brighten = Slider("중간 밝기를 올리고 채도를 살짝 높임")
        self.sharpen = Slider("윤곽을 또렷하게")
        self.grain = Slider("미세한 입자로 자연스러운 질감")
        self.clarity = Slider("넓은 범위의 밝기 대비를 키워 입체감을 줍니다 (사진 앱의 '명료도')")
        self.vibrance = Slider("채도 낮은 색 위주로 생생하게 (피부·진한 색은 덜 과해짐)")
        f.addRow("스타일 프리셋", self.style)
        f.addRow("피부 보정", self.soft_skin)
        f.addRow("빛 번짐", self.glow)
        f.addRow("화사함", self.brighten)
        f.addRow("선명도", self.sharpen)
        f.addRow("명료도 (입체감)", self.clarity)
        f.addRow("생생함", self.vibrance)
        f.addRow("필름 그레인", self.grain)
        self.style_filter_note = note("필터만 모드에서는 피부 보정 · 빛 번짐 · 필름 그레인을 쓰지 않습니다.")
        f.addRow(self.style_filter_note)
        self.style.activated.connect(lambda _: self._on_preset(self.style, pipeline.STYLE_PRESETS))
        for s in (self.soft_skin, self.glow, self.brighten, self.sharpen, self.grain, self.clarity, self.vibrance):
            s.changed.connect(self._changed)
        return page

    def _tab_output(self):
        page, f = self._form_tab()
        self.codec = _combo([("H.265 / HEVC (추천)", "h265"), ("H.264 (호환성 최고)", "h264"),
                             ("AV1 (압축률 최고)", "av1")])
        self.encoder = _combo(ENCODER_ITEMS, "고른 하드웨어가 없으면 자동으로 바뀝니다")
        self.crf = QSpinBox()
        self.crf.setRange(10, 35)
        self.crf.setToolTip("낮을수록 고화질·큰 용량 (18~22 추천)")
        self.ten_bit = Toggle("10비트 (그라데이션 계단 현상 감소)")
        self.container = _combo([("MP4", ".mp4"), ("MKV", ".mkv")])
        self.audio = _combo(AUDIO_ITEMS)
        self.chunk = QSpinBox()
        self.chunk.setRange(10, 600)
        self.chunk.setSuffix(" 초")
        self.chunk.setToolTip("이 간격마다 저장해 두어, 중간에 멈춰도 이어서 처리합니다")
        self.open_after = Toggle("완료 후 결과 폴더 열기")
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("비워 두면 원본과 같은 폴더")
        b = button("찾기", "folder")
        b.clicked.connect(self._pick_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(b)
        f.addRow("코덱", self.codec)
        f.addRow("인코더", self.encoder)
        f.addRow("화질 (CRF)", self.crf)
        f.addRow(self.ten_bit)
        f.addRow("파일 형식", self.container)
        f.addRow("오디오", self.audio)
        f.addRow("이어하기 저장 간격", self.chunk)
        f.addRow("저장 위치", out_row)
        f.addRow(self.open_after)
        for c in (self.codec, self.encoder, self.audio):
            c.currentIndexChanged.connect(self._changed)
        self.crf.valueChanged.connect(self._changed)
        self.chunk.valueChanged.connect(self._changed)
        self.ten_bit.toggled.connect(self._changed)
        self.container.currentIndexChanged.connect(lambda _: self._save_cfg())
        self.open_after.toggled.connect(lambda _: self._save_cfg())
        self.out_dir.editingFinished.connect(self._save_cfg)
        return page

    def _build_right(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 4, 8)
        self.view = CompareView()
        lay.addWidget(self.view, 1)

        row = QHBoxLayout()
        lbl = QLabel("장면 위치")
        lbl.setObjectName("muted")
        row.addWidget(lbl)
        self.time = QSlider(Qt.Orientation.Horizontal)
        self.time.setRange(0, 1000)
        self.time.setValue(300)
        self.time_lbl = QLabel("0:00")
        self.time_lbl.setMinimumWidth(40)
        self.time.valueChanged.connect(self._update_time_label)
        row.addWidget(self.time, 1)
        row.addWidget(self.time_lbl)
        self.btn_preview = button("미리보기", "eye")
        self.btn_preview.clicked.connect(self._preview)
        self.btn_sample = button("10초 샘플", "film", tip="지금 장면 위치부터 10초를 실제로 변환해 봅니다")
        self.btn_sample.clicked.connect(self._sample)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_sample)
        lay.addLayout(row)

        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        bottom = QHBoxLayout()
        self.status = QLabel("준비됨")
        self.status.setObjectName("muted")
        bottom.addWidget(self.status, 1)
        self.btn_open = button("결과 폴더", "folder")
        self.btn_open.clicked.connect(self._open_out)
        self.btn_start = button("변환 시작", "play", "primary")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = button("중지", "stop", "danger")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        bottom.addWidget(self.btn_open)
        bottom.addWidget(self.btn_stop)
        bottom.addWidget(self.btn_start)
        lay.addLayout(bottom)
        return w

    # ── 설정 ↔ 화면 ──
    def settings(self):
        s = pipeline.Settings()
        res = self.res.currentData()
        s.target_height = self.res_custom.value() if res == -1 else res
        s.scale = 1.0
        fps = self.fps.currentData()
        s.fps = f"{self.fps_custom.value():g}" if fps == "custom" else fps
        s.device_sr = self.device.currentData()
        mode = self.mode.currentData()
        s.engine = "filter" if mode == "filter" else "ai"
        s.ai_restore = mode == "ai"
        s.sr_model = self.sr_model.currentData() or "general_v3"
        s.sr_int8 = self.sr_int8.isChecked()
        s.proc_scale = self.proc_scale.currentData()
        s.denoise = round(self.denoise.value(), 2)
        s.face_enhance = self.face_enhance.isChecked()
        s.face_strength = round(self.face_strength.value(), 2)
        s.face_min = self.face_min.value()
        s.face_max = self.face_max.value()
        for k in ("texture", "soft_skin", "glow", "brighten", "sharpen", "grain", "clarity", "vibrance"):
            setattr(s, k, round(getattr(self, k).value(), 2))
        s.deinterlace = self.deinterlace.currentData()
        s.rife_model = self.rife_model.currentData()
        sc = self.scene.value()
        s.scene_threshold = round(0.45 - 0.4 * sc, 3) if sc > 0 else 1.0
        s.dedup = self.dedup.isChecked()
        s.interp_res = self.interp_res.currentData()
        s.device_rife = self.device_rife.currentData()
        s.codec = self.codec.currentData()
        s.encoder = self.encoder.currentData()
        s.crf = self.crf.value()
        s.ten_bit = self.ten_bit.isChecked()
        s.audio = self.audio.currentData()
        s.chunk_seconds = self.chunk.value()
        return s

    def _apply_settings(self, s):
        # 예전 버전은 장치 ID 대신 표시 이름("CPU (OpenVINO)" 등)을 저장했다 → ID 로 바꿔서 읽는다
        ids = {name: dev for dev, name in backends.list_devices()}
        s.device_sr = ids.get(s.device_sr, s.device_sr)
        s.device_rife = ids.get(s.device_rife, s.device_rife)
        self._loading = True
        try:
            if s.target_height and not _set_combo(self.res, s.target_height):
                _set_combo(self.res, -1)
            elif not s.target_height:
                _set_combo(self.res, 0)
            self.res_custom.setValue(s.target_height or 1080)
            if not _set_combo(self.fps, s.fps):
                _set_combo(self.fps, "custom")
            try:
                self.fps_custom.setValue(float(s.fps))
            except ValueError:
                self.fps_custom.setValue(60)
            _set_combo(self.device, s.device_sr)
            _set_combo(self.mode, "filter" if s.engine == "filter" else "ai" if s.ai_restore else "interp")
            _set_combo(self.sr_model, s.sr_model)
            self.sr_int8.setChecked(s.sr_int8 and esrgan.int8_available())
            _set_combo(self.proc_scale, float(s.proc_scale))
            self.denoise.setValue(s.denoise)
            self.face_enhance.setChecked(s.face_enhance and face_mod.available())
            self.face_strength.setValue(s.face_strength)
            self.face_min.setValue(s.face_min)
            self.face_max.setValue(s.face_max)
            for k in ("texture", "soft_skin", "glow", "brighten", "sharpen", "grain", "clarity", "vibrance"):
                getattr(self, k).setValue(getattr(s, k))
            _set_combo(self.deinterlace, s.deinterlace)
            _set_combo(self.rife_model, s.rife_model)
            self.scene.setValue(0 if s.scene_threshold >= 1 else (0.45 - s.scene_threshold) / 0.4)
            self.dedup.setChecked(s.dedup)
            _set_combo(self.interp_res, s.interp_res)
            _set_combo(self.device_rife, s.device_rife)
            _set_combo(self.codec, s.codec)
            _set_combo(self.encoder, s.encoder)
            self.crf.setValue(s.crf)
            self.ten_bit.setChecked(s.ten_bit)
            _set_combo(self.audio, s.audio)
            self.chunk.setValue(s.chunk_seconds)
        finally:
            self._loading = False
        self._sync_enabled()
        self._sync_preset_labels()

    def _sync_enabled(self):
        self.res_custom.setVisible(self.res.currentData() == -1)
        self.fps_custom.setVisible(self.fps.currentData() == "custom")
        mode = self.mode.currentData()
        ai, filt = mode == "ai", mode == "filter"
        self.mode_desc.setText({
            "ai": "AI 가 압축 깨짐을 지우고 디테일·얼굴을 복원합니다 (가장 좋은 화질 · GPU·NPU·CPU 를 함께 사용).",
            "interp": "AI 화질 복원 없이 단순 확대 + 프레임 보간만 합니다 (60fps 변환용 · 보간은 CPU·GPU 가 나눠 처리).",
            "filter": "AI 없이 FFmpeg 필터로 밝고 깨끗하게 (실시간에 가까움 · 원본 프레임레이트). 필터용 값이 자동으로 "
                      "들어가고, 다른 방식으로 바꾸면 원래 값으로 돌아갑니다.",
        }[mode])
        # AI 복원 전용: 품질/속도 프리셋 · AI 화질 복원 · 얼굴 복원
        self.quality.setVisible(ai)
        self.quality_label.setVisible(ai)
        for card in (self.sr_card, self.face_card):
            card.setVisible(ai)
            card.sync()
        self.sr_int8.setEnabled(esrgan.int8_available() and self.sr_model.currentData() == "general_v3")
        # 압축 노이즈 제거: 필터만은 FFmpeg 노이즈 제거, AI 복원은 General v3 에서만, 보간만은 쓰지 않음
        self.denoise.setEnabled(filt or (ai and self.sr_model.currentData() == "general_v3"))
        self.common_card.desc.setText("보간만 모드에서는 압축 노이즈 제거를 쓰지 않습니다" if mode == "interp"
                                      else "AI 복원과 필터만에 적용됩니다")
        # 보간 탭: 보간을 하지 않는 설정이면 막고 이유를 보여 준다
        fps = self.fps.currentData()
        _, info = self._current()
        if filt:
            why = "필터만 모드는 보간 없이 원본 프레임레이트로 저장합니다 ('복원' 탭)."
        elif fps == "src":
            why = "프레임레이트가 '원본 그대로' 라 보간하지 않습니다 ('기본' 탭에서 바꿀 수 있음)."
        elif info is not None and not pipeline.Plan(info, self.settings()).interp:
            why = "출력 프레임레이트가 원본과 같아 보간하지 않습니다."
        else:
            why = ""
        self.interp_off.setText(why)
        self.interp_off.setVisible(bool(why))
        for wdg in (self.rife_model, self.scene, self.dedup, self.interp_res):
            wdg.setEnabled(not why)
        # 연산 장치: 업스케일·얼굴은 AI 복원에서만, 보간은 필터만이 아닐 때 쓴다
        # (필터만은 FFmpeg 필터를 CPU 로, 인코딩은 하드웨어 인코더로 처리)
        self.device.setEnabled(ai)
        self.device_rife.setEnabled(not filt)
        self.dev_card.desc.setText("필터만 모드에서는 쓰지 않습니다 (필터는 CPU, 인코딩은 하드웨어 인코더)" if filt else
                                   "보간만 모드에서는 보간 장치만 씁니다 (자동: CPU·GPU 가 나눠 처리)" if not ai else
                                   "보통은 '자동' 그대로 두세요")
        # 스타일 탭: 필터만 모드에서 쓰지 않는 값
        for wdg in (self.soft_skin, self.glow, self.grain):
            wdg.setEnabled(not filt)
        self.style_filter_note.setVisible(filt)

    def _on_engine(self, _):
        """처리 방식을 '필터만' 으로 바꾸면 필터용 값(FILTER_LOOK)을 넣고, 다른 방식으로 돌아오면 되돌린다."""
        if self._loading:
            return
        s = self.settings()
        if s.engine == "filter":
            if self._before_filter is None:
                self._before_filter = {k: getattr(s, k) for k in pipeline.FILTER_LOOK}
            for k, v in pipeline.FILTER_LOOK.items():
                setattr(s, k, v)
        elif self._before_filter:
            for k, v in self._before_filter.items():
                setattr(s, k, v)
            self._before_filter = None
        self._apply_settings(s)
        self._changed()

    def _on_preset(self, combo, presets):
        name = combo.currentData()
        if name in presets:
            s = self.settings()
            # 이전 프리셋이 추가로 바꿨던 항목(예: 음악방송 프리셋의 얼굴 수 제한·경량화)은
            # 새 프리셋이 다루지 않으면 기본값으로 되돌린다 → 기존 프리셋은 예전과 똑같이 동작
            prev = self._last_preset.get(id(combo))
            if prev in presets and prev != name:
                default = pipeline.Settings()
                # 출력 해상도·프레임레이트는 사용자가 따로 정하는 항목이라 되돌리지 않는다
                for k in set(presets[prev]) - set(presets[name]) - OUTPUT_KEYS:
                    setattr(s, k, getattr(default, k))
            for k, v in presets[name].items():
                setattr(s, k, v)
            self._apply_settings(s)
            self._last_preset[id(combo)] = name
        self._changed()

    def _sync_preset_labels(self):
        s = self.settings()

        def same(k, v):
            cur = getattr(s, k)
            return abs(float(cur) - float(v)) < 0.02 if isinstance(v, (int, float)) else cur == v

        for combo, presets in ((self.quality, pipeline.QUALITY_PRESETS), (self.style, pipeline.STYLE_PRESETS)):
            # 출력 해상도·프레임레이트는 품질/속도와 따로 정하는 항목 → 바꿔도 프리셋 표시는 그대로
            checked = {n: {k: v for k, v in d.items() if k not in OUTPUT_KEYS} for n, d in presets.items()}
            # 여러 프리셋에 해당하면 가장 구체적인(항목이 많은) 프리셋을 표시
            # (예: '음악방송 4K 60fps (추천)' 은 '일반 · 빠름' 의 값을 모두 포함한다)
            matches = [(len(d), n) for n, d in checked.items() if all(same(k, v) for k, v in d.items())]
            match = max(matches)[1] if matches else CUSTOM
            combo.blockSignals(True)
            _set_combo(combo, match)
            combo.blockSignals(False)
            # 불러온 설정·파일별 설정이 어느 프리셋 상태인지도 기억해 둔다
            # (음악방송 프리셋 상태로 시작한 뒤 다른 프리셋을 골라도, 그 프리셋이 바꾼 항목이 원래대로 돌아가도록)
            if match in presets:
                self._last_preset[id(combo)] = match

    def _changed(self, *_):
        if self._loading:
            return
        self._sync_enabled()
        self._sync_preset_labels()
        it = self.files.currentItem()
        if it is not None:
            it.setData(ROLE_SET, self.settings().to_dict())
        self._update_info()

    # ── 파일 ──
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        """끌어다 놓은 파일을 종류별로 나눠 영상은 영상 탭, 사진은 사진 탭으로 보낸다."""
        files = []
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            files += sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p]
        videos = [str(p) for p in files if p.suffix.lower() in VIDEO_EXT]
        photos = [str(p) for p in files if p.suffix.lower() in IMAGE_EXT]
        if photos:
            self.photo.add_files(photos)
            self.main_tabs.setCurrentWidget(self.photo)
        if videos:
            self._add_files(videos)
            self.main_tabs.setCurrentIndex(0)

    def _add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "영상 선택", self.cfg.get("last_dir", ""),
                                                "영상 (" + " ".join("*" + e for e in VIDEO_EXT) + ")")
        if paths:
            self.cfg["last_dir"] = str(Path(paths[0]).parent)
        self._add_files(paths)

    def _add_files(self, paths):
        existing = {self.files.item(i).data(ROLE_PATH) for i in range(self.files.count())}
        added = []
        base = self.settings().to_dict()
        for p in paths:
            if p in existing:
                continue
            try:
                info = video_io.probe(p)
            except Exception as e:
                QMessageBox.warning(self, "열 수 없음", f"{p}\n\n{e}")
                continue
            m, sec = divmod(int(info.duration), 60)
            it = QListWidgetItem(f"{Path(p).name}   ({info.width}x{info.height} · {info.fps_float:.2f}fps · "
                                 f"{m}:{sec:02d})")
            it.setData(ROLE_PATH, p)
            it.setData(ROLE_INFO, info)
            it.setData(ROLE_SET, dict(base))
            self.files.addItem(it)
            added.append(p)
        if self.files.currentRow() < 0 and self.files.count():
            self.files.setCurrentRow(0)
        if added and self.auto_analyze.isChecked():
            self._analyze(added)

    def _remove_selected(self):
        for it in self.files.selectedItems():
            self.files.takeItem(self.files.row(it))
        self._on_row_changed(self.files.currentRow())

    def _clear_files(self):
        self.files.clear()
        self._on_row_changed(-1)

    def _find_item(self, path):
        for i in range(self.files.count()):
            if self.files.item(i).data(ROLE_PATH) == path:
                return self.files.item(i)
        return None

    def _on_row_changed(self, row):
        it = self.files.item(row) if row >= 0 else None
        self._before_filter = None  # 되돌릴 값은 그 파일에서 처리 방식을 바꿀 때만 의미가 있다
        if it is not None:
            self._apply_settings(pipeline.Settings.from_dict(it.data(ROLE_SET)))
        self._show_analysis(it)
        self._update_info()

    def _current(self):
        it = self.files.currentItem()
        if it is None:
            return None, None
        return it.data(ROLE_PATH), it.data(ROLE_INFO)

    # ── 분석 · 추천 · 기본값 ──
    def _analyze(self, paths, force_apply=True):
        if self.thread is not None:
            self.status.setText("다른 작업 중이라 분석을 건너뛰었습니다. 나중에 [추천값 적용] 을 누르세요.")
            return
        jobs = []
        for p in paths:
            it = self._find_item(p)
            if it is not None:
                jobs.append((p, it.data(ROLE_INFO), pipeline.Settings.from_dict(it.data(ROLE_SET))))
        results = {}

        def job(w):
            for n, (p, info, cur) in enumerate(jobs, 1):
                w.progress.emit(pipeline.Progress(n - 1, len(jobs), 0, 0, f"영상 분석 중… {Path(p).name}"), 1, 1)
                results[p] = analyze.recommend(info, cur)
            return f"분석 완료 ({len(jobs)}개)"

        def done(_):
            changes = []
            cur_path = self.files.currentItem().data(ROLE_PATH) if self.files.currentItem() else None
            for p, a in results.items():
                it = self._find_item(p)
                if it is None:
                    continue
                it.setData(ROLE_RECO, a.settings)
                it.setData(ROLE_WHY, a.reasons)
                if force_apply:
                    old = dict(it.data(ROLE_SET))
                    merged = dict(old)
                    merged.update(a.settings)
                    it.setData(ROLE_SET, merged)
                    if p == cur_path:
                        changes = _describe_changes(old, merged)
            self._on_row_changed(self.files.currentRow())
            if force_apply and cur_path in results:
                if changes:
                    self.status.setText("추천값 적용 — 바뀐 항목: " + ", ".join(changes))
                    self.analysis.setText(self.analysis.text() + "<br><br><b>이번에 바뀐 항목</b><br>"
                                          + "<br>".join("• " + c for c in changes))
                else:
                    self.status.setText("추천값이 지금 설정과 같습니다 (바뀐 항목 없음)")

        self.analysis.setText("<i>영상 분석 중…</i>")
        self._run(job, done)

    def _apply_recommendation(self):
        it = self.files.currentItem()
        if it is None:
            QMessageBox.information(self, "알림", "먼저 영상을 추가하고 선택하세요.")
            return
        self._changed()  # 화면의 최신 값을 먼저 저장해 두고
        self._analyze([it.data(ROLE_PATH)])  # 매번 새로 분석해서 적용 (바뀐 항목을 알려 줌)

    def _reset_defaults(self):
        self._apply_settings(pipeline.Settings())
        self._changed()
        self.status.setText("기본값으로 되돌렸습니다")

    def _apply_to_all(self):
        d = self.settings().to_dict()
        for i in range(self.files.count()):
            self.files.item(i).setData(ROLE_SET, dict(d))
        self.status.setText(f"지금 설정을 {self.files.count()}개 파일 모두에 적용했습니다")

    def _show_analysis(self, it):
        if it is None:
            self.analysis.setText("영상을 추가하면 압축 상태·해상도·프레임을 분석해 알맞은 설정을 추천합니다.")
            return
        why = it.data(ROLE_WHY)
        if not why:
            self.analysis.setText("아직 분석하지 않았습니다. [추천값 적용] 을 누르면 분석합니다.")
            return
        self.analysis.setText("<br>".join("• " + w for w in why))

    # ── 정보 표시 ──
    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더", self.out_dir.text())
        if d:
            self.out_dir.setText(d)
            self._save_cfg()

    def _output_path(self, path, s, info, suffix=""):
        name = output_name(path, s, info, suffix, self.container.currentData())
        folder = Path(self.out_dir.text()) if self.out_dir.text().strip() else Path(path).parent
        out, n = folder / name.name, 1
        while out.exists():
            out = folder / f"{name.stem} ({n}){name.suffix}"
            n += 1
        return str(out)

    def _update_time_label(self):
        _, info = self._current()
        if info:
            t = info.duration * self.time.value() / 1000
            self.time_lbl.setText(f"{int(t // 60)}:{int(t % 60):02d}")

    def _update_info(self):
        self._update_time_label()
        path, info = self._current()
        if not info:
            self.info.setText("영상을 추가하세요. 폴더를 끌어다 놓으면 안의 영상이 모두 추가됩니다.")
            return
        s = self.settings()
        if s.engine == "filter":
            plan = pipeline.Plan(info, pipeline.Settings.from_dict(dict(s.to_dict(), fps="src")))
            self.info.setText(f"{info.width}x{info.height} {info.fps_float:.2f}fps → <b>{plan.ow}x{plan.oh} "
                              f"{info.fps_float:.2f}fps</b><br>필터만 (AI·보간·얼굴 복원 없음) — "
                              "원본 프레임레이트로 저장, 매우 빠름")
            return
        plan = pipeline.Plan(info, s)
        sr, rife = pipeline.resolve_devices(s)
        out_n = plan.out_frames(info.frames)
        rifes = pipeline.interp_devices(s, rife) if plan.interp else []
        devs = (f"업스케일 {sr if s.ai_restore else '없음(단순 확대)'} · "
                f"보간 {' + '.join(rifes) if rifes else '없음'}")
        if plan.faces:
            fdev = pipeline.resolve_face_device(s, sr)
            extra = pipeline.extra_face_device(fdev)
            devs += f" · 얼굴 {fdev}{' + ' + extra if extra else ''}"
        text = (f"{info.width}x{info.height} {info.fps_float:.2f}fps → <b>{plan.ow}x{plan.oh} "
                f"{float(plan.fout):.2f}fps</b><br>AI 처리 {plan.pw}x{plan.ph} · 출력 {out_n:,}프레임"
                f"{' · 인터레이스 제거' if plan.deint else ''}<br>장치: {devs}")
        if plan.ai_short:
            text += ("<br><span style='color:#f0b35a'>⚠ 출력이 처리 해상도의 4배를 넘어 일부는 단순 확대됩니다. "
                     "'복원' 탭에서 처리 해상도를 올리면 더 선명해집니다.</span>")
        speed = self.cfg.get("speed", {}).get(self._speed_key(s, plan))
        if speed:
            eta = out_n / speed
            if eta < 60:
                when = "1분 미만"
            elif eta < 3600:
                when = f"약 {int(eta // 60)}분"
            else:
                when = f"약 {int(eta // 3600)}시간 {int(eta % 3600 // 60)}분"
            text += f"<br>예상 시간: {when} (이전 기록 기준)"
        self.info.setText(text)

    def _speed_key(self, s, plan):
        sr, _ = pipeline.resolve_devices(s)
        return f"{sr}|{plan.pw}x{plan.ph}|{plan.ow}x{plan.oh}|{plan.interp}|{s.ai_restore}"

    # ── 작업 실행 ──
    def _run(self, fn, on_done=None):
        if self.thread is not None:
            return False
        self.stop_event = threading.Event()
        self._on_done = on_done
        self.thread = QThread()
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # 반드시 메인 창의 메서드(슬롯)에 연결해야 화면 스레드에서 실행된다.
        # (람다로 연결하면 작업 스레드에서 실행되어 완료 시 프로그램이 강제 종료됨)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview.connect(self.view.set_images)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._busy(True)
        self.thread.start()
        return True

    def _cleanup(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None
        self._busy(False)

    @Slot(str)
    def _on_finished(self, msg):
        on_done, self._on_done = self._on_done, None
        self._cleanup()
        self.status.setText(msg or "완료")
        if on_done and msg != "중지됨":
            try:
                on_done(msg)
            except Exception:
                log.error("완료 처리 실패\n%s", traceback.format_exc())

    @Slot(str)
    def _on_failed(self, msg):
        self._on_done = None
        self._cleanup()
        self.status.setText("오류")
        self._show_analysis(self.files.currentItem())
        QMessageBox.critical(self, "오류", msg[-3000:])

    def _busy(self, on):
        for wdg in (self.btn_start, self.btn_preview, self.btn_sample, self.btn_reco):
            wdg.setEnabled(not on)
        self.btn_stop.setEnabled(on)

    def _preview(self):
        path, info = self._current()
        if not path:
            QMessageBox.information(self, "알림", "먼저 영상을 추가하고 선택하세요.")
            return
        s = self.settings()
        t = info.duration * self.time.value() / 1000
        self.status.setText("미리보기 처리 중… (처음에는 모델 준비로 조금 걸립니다)")
        self.view.set_message("처리 중…")

        def job(w):
            t0 = time.time()
            before, after = pipeline.preview_frame(path, s, t)
            w.preview.emit(before, after)
            return f"미리보기 완료 ({time.time() - t0:.1f}초) — {after.shape[1]}x{after.shape[0]}"
        self._run(job)

    def _sample(self):
        path, info = self._current()
        if not path:
            QMessageBox.information(self, "알림", "먼저 영상을 추가하고 선택하세요.")
            return
        s = self.settings()
        start = max(0.0, min(info.duration - 10, info.duration * self.time.value() / 1000))
        out = self._output_path(path, s, info, "_샘플")

        def job(w):
            pipeline.process(path, out, s, lambda p: w.progress.emit(p, 1, 1), self.stop_event,
                             start=start, duration=10)
            return f"샘플 저장: {out}"
        self._run(job, lambda _: QDesktopServices.openUrl(QUrl.fromLocalFile(out)))

    def _start(self):
        if not self.files.count():
            QMessageBox.information(self, "알림", "먼저 영상을 추가하세요.")
            return
        self._changed()
        jobs = []  # 화면 값은 모두 여기(메인 스레드)에서 읽어 둔다
        for i in range(self.files.count()):
            it = self.files.item(i)
            path, info = it.data(ROLE_PATH), it.data(ROLE_INFO)
            s = pipeline.Settings.from_dict(it.data(ROLE_SET))
            jobs.append((path, info, s, self._output_path(path, s, info), self._speed_key(s, pipeline.Plan(info, s))))
        self._save_cfg()
        speeds = {}
        open_after = self.open_after.isChecked()

        def job(w):
            for n, (path, info, s, out, key) in enumerate(jobs, 1):
                t0 = time.time()
                pipeline.process(path, out, s, lambda p, n=n: w.progress.emit(p, n, len(jobs)), self.stop_event)
                speeds[key] = round(pipeline.Plan(info, s).out_frames(info.frames) / max(1, time.time() - t0), 3)
            return f"완료: {len(jobs)}개 파일 저장됨"

        def done(_):
            self.cfg.setdefault("speed", {}).update(speeds)
            self._save_cfg()
            self._update_info()
            if open_after:
                self._open_out()
        self._run(job, done)

    def _stop(self):
        self.stop_event.set()
        self.status.setText("중지하는 중… (완료된 부분은 저장되어 다음에 이어서 처리됩니다)")

    @Slot(object, int, int)
    def _on_progress(self, p, n, total):
        self.bar.setMaximum(max(1, p.total))
        self.bar.setValue(p.done)
        prefix = f"[{n}/{total}] " if total > 1 else ""
        if p.stage == "처리 중":
            eta = int(p.eta)
            self.status.setText(f"{prefix}{p.done:,}/{p.total:,} 프레임 · {p.fps:.2f} fps · "
                                f"남은 시간 {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}")
        else:
            self.status.setText(prefix + p.stage)
        if p.frame is not None:
            self.view.set_images(None, p.frame)

    def _open_out(self):
        d = self.out_dir.text().strip()
        if not d:
            path, _ = self._current()
            d = str(Path(path).parent) if path else ""
        if d:
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    # ── 프리셋 / 설정 저장 ──
    def _reload_presets(self):
        self.my_presets.clear()
        for p in sorted(PRESETS.glob("*.json")):
            self.my_presets.addItem(p.stem, str(p))

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "프리셋 저장", "이름:")
        if ok and name.strip():
            safe = "".join(c for c in name.strip() if c not in '\\/:*?"<>|')
            (PRESETS / f"{safe}.json").write_text(
                json.dumps(self.settings().to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._reload_presets()
            _set_combo(self.my_presets, str(PRESETS / f"{safe}.json"))

    def _load_preset(self):
        p = self.my_presets.currentData()
        if p:
            self._apply_settings(pipeline.Settings.from_dict(json.loads(Path(p).read_text(encoding="utf-8"))))
            self._changed()

    def _delete_preset(self):
        p = self.my_presets.currentData()
        if p and QMessageBox.question(self, "삭제", f"'{Path(p).stem}' 프리셋을 삭제할까요?") \
                == QMessageBox.StandardButton.Yes:
            Path(p).unlink(missing_ok=True)
            self._reload_presets()

    def _load_cfg(self):
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cfg(self):
        self.cfg.update(settings=self.settings().to_dict(), out_dir=self.out_dir.text(),
                        container=self.container.currentData(), auto_analyze=self.auto_analyze.isChecked(),
                        open_after=self.open_after.isChecked())
        try:
            CONFIG.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass  # 읽기 전용 위치에서 실행해도 동작은 하도록

    def _about(self):
        QMessageBox.about(
            self, "정보",
            f"<b>AI 영상·사진 업스케일러</b> v{__version__}<br><br>"
            "Copyright © 2026 최승훈. All rights reserved.<br><br>"
            "소스 코드는 공개되어 있지만 오픈소스가 아니며, 저작권자의 서면 허락 없이<br>"
            "수정·재배포·상업적 이용을 할 수 없습니다. 자세한 내용은 "
            "<i>LICENSE.txt</i> 를 참고하세요.<br><br>"
            "함께 배포되는 오픈소스 구성 요소(FFmpeg, Qt/PySide6, OpenVINO, OpenCV, AI 모델 등)는<br>"
            "각자의 라이선스를 따르며 위 제한을 받지 않습니다 — <i>THIRD_PARTY_NOTICES.txt</i> 참고.")

    def closeEvent(self, e):
        if self.thread is not None:
            if QMessageBox.question(self, "종료", "처리 중입니다. 중지하고 종료할까요?\n"
                                    "(완료된 부분은 다음에 이어서 처리됩니다)") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.stop_event.set()
            self.thread.quit()
            self.thread.wait(15000)
        self._save_cfg()
        e.accept()


def _setup_logging():
    """pythonw 로 실행하면 콘솔이 없으므로 출력과 오류를 cache\\error.log 에 남긴다."""
    CACHE.mkdir(exist_ok=True)
    log_file = open(CACHE / "error.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log_file
    if sys.stderr is None:
        sys.stderr = log_file
    faulthandler.enable(log_file)  # 네이티브 충돌도 기록
    logging.basicConfig(stream=log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if os.environ.get("VUP_DEBUG"):
        # 디버그 모드(콘솔 실행): 오류를 콘솔에도 바로 보여 준다 (화면을 그대로 Claude 에게 전달하면 됨)
        console = logging.StreamHandler(sys.__stderr__ or sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(console)
        logging.getLogger().setLevel(logging.DEBUG)
        log.info("디버그 모드 시작: v%s, 프로그램 폴더 %s", __version__, CACHE.parent)

    def hook(exc_type, exc, tb):
        log.error("처리되지 않은 오류\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(None, "오류", f"{exc}\n\n(자세한 내용: cache\\error.log)")
    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def main():
    _setup_logging()
    app = QApplication(sys.argv)
    app.setFont(QFont("Malgun Gothic", 9) if sys.platform == "win32" else app.font())
    ui.apply_theme(app)
    app.setWindowIcon(ui.app_icon())
    w = MainWindow()
    files = [a for a in sys.argv[1:] if os.path.isfile(a)]
    w.show()
    if files:
        w._add_files(files)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
