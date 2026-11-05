"""화면 디자인: 테마 · 아이콘 · 토글 스위치 · 세그먼트 버튼 · 기능 카드.

아이콘은 이 프로그램용으로 직접 그린 단순한 선 아이콘(SVG)이다 (외부 아이콘 세트를 쓰지 않음).
"""
import functools

from PySide6.QtCore import (QByteArray, QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt,
                            Property, Signal)
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QListWidget,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from .paths import CACHE

# ── 색 ────────────────────────────────────────────────────────────────────
BG = "#0e1014"        # 창 바탕
SURFACE = "#151820"   # 섹션
CARD = "#1b1f29"      # 카드
RAISED = "#232836"    # 버튼 · 선택된 탭
FIELD = "#11141a"     # 입력칸
BORDER = "#2a3040"
BORDER_HI = "#3a4258"
TEXT = "#e9ecf2"
MUTED = "#8e97aa"
FAINT = "#5d6577"
ACCENT = "#6c7cff"
ACCENT2 = "#9b6bff"
DANGER = "#ff7a88"
OK = "#3ecf8e"

# ── 아이콘 (24x24 선 아이콘, stroke 는 그릴 때 색을 넣는다) ─────────────────
_ICONS = {
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "trash": '<path d="M4 7h16M9.5 7V4.5h5V7M6.5 7l1 12.5h9l1-12.5M10 11v5M14 11v5"/>',
    "play": '<path d="M8 5.5v13l10.5-6.5z" fill="{c}"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="{c}"/>',
    "folder": '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>'
           '<circle cx="12" cy="12" r="3"/>',
    "film": '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M8 4.5v15M16 4.5v15M3.5 9h4.5M3.5 15h4.5'
            'M16 9h4.5M16 15h4.5"/>',
    "sparkles": '<path d="M10 3.5c.6 4 2.5 5.9 6.5 6.5-4 .6-5.9 2.5-6.5 6.5-.6-4-2.5-5.9-6.5-6.5 4-.6 5.9-2.5 6.5-6.5z"/>'
                '<path d="M18 14.5c.3 1.8 1.2 2.7 3 3-1.8.3-2.7 1.2-3 3-.3-1.8-1.2-2.7-3-3 1.8-.3 2.7-1.2 3-3z"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 2.5-5.8"/><path d="M4 4.5v4.5h4.5"/>',
    "copy": '<rect x="8.5" y="8.5" width="12" height="12" rx="2.5"/>'
            '<path d="M15.5 8.5V6a2.5 2.5 0 0 0-2.5-2.5H6A2.5 2.5 0 0 0 3.5 6v7A2.5 2.5 0 0 0 6 15.5h2.5"/>',
    "sliders": '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    "frames": '<rect x="2.5" y="6" width="11" height="12" rx="2"/><path d="M16.5 6.5h1.5a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-1.5"/>'
              '<path d="M21.5 9v6"/>',
    "palette": '<path d="M12 3.5a8.5 8.5 0 1 0 0 17c1.2 0 1.7-.9 1.3-1.8-.5-1.1.3-2.4 1.6-2.4h1.8A3.8 3.8 0 0 0 20.5 12.5'
               'C20.5 7.5 16.7 3.5 12 3.5z"/><circle cx="8" cy="11" r="1" fill="{c}"/><circle cx="11" cy="7.5" r="1" '
               'fill="{c}"/><circle cx="15.5" cy="8.5" r="1" fill="{c}"/>',
    "export": '<path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M4.5 19.5h15"/>',
    "video": '<rect x="3" y="6" width="13" height="12" rx="2.5"/><path d="M16 10.5l5-3v9l-5-3z"/>',
    "image": '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><circle cx="9" cy="10" r="1.6"/>'
             '<path d="M20.5 15.5l-4.5-4.5-8.5 8.5"/>',
    "face": '<circle cx="12" cy="12" r="8.5"/><path d="M9 10h.01M15 10h.01"/><path d="M8.8 14.3c.9 1.2 2 1.8 3.2 1.8'
            's2.3-.6 3.2-1.8"/>',
    "bolt": '<path d="M13 3 5 13.5h6L10 21l8-10.5h-6z"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8h.01"/>',
    "download": '<path d="M12 4v10M7.5 9.5 12 14l4.5-4.5M5 19.5h14"/>',
    "save": '<path d="M5.5 3.5h10l3 3v12a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2z"/>'
            '<path d="M8 3.5v4.5h7M8 20.5v-6h8v6"/>',
    "upload": '<path d="M12 15.5V4.5M7.5 9 12 4.5 16.5 9M4.5 15.5v2a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2"/>',
    "chevron-down": '<path d="M7 10l5 5 5-5"/>',
    "chevron-up": '<path d="M7 14l5-5 5 5"/>',
    "layers": '<path d="M12 4 3.5 8.5 12 13l8.5-4.5z"/><path d="M3.5 12.5 12 17l8.5-4.5"/>',
}


def _svg(name, color, stroke=1.8):
    body = _ICONS[name].replace("{c}", color)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def _render(svg, size, ratio=2.0):
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    QSvgRenderer(QByteArray(svg.encode())).render(p)
    p.end()
    pm.setDevicePixelRatio(ratio)
    return pm


@functools.lru_cache(maxsize=None)
def icon(name, color=TEXT, size=18):
    ic = QIcon()
    ic.addPixmap(_render(_svg(name, color), size), QIcon.Mode.Normal)
    ic.addPixmap(_render(_svg(name, FAINT), size), QIcon.Mode.Disabled)
    return ic


def pixmap(name, color=TEXT, size=18):
    return _render(_svg(name, color), size)


# ── 프로그램 아이콘: 보라 그라데이션 둥근 사각형 + 반짝임(AI) ──
APP_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#5b6cff"/><stop offset="0.55" stop-color="#8b5cf6"/>'
    '<stop offset="1" stop-color="#d946ef"/></linearGradient>'
    '<radialGradient id="h" cx="0.3" cy="0.2" r="0.8"><stop offset="0" stop-color="#ffffff" stop-opacity="0.28"/>'
    '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></radialGradient></defs>'
    '<rect x="8" y="8" width="240" height="240" rx="64" fill="url(#g)"/>'
    '<rect x="8" y="8" width="240" height="240" rx="64" fill="url(#h)"/>'
    '<path d="M112 52c6 40 24 58 64 64-40 6-58 24-64 64-6-40-24-58-64-64 40-6 58-24 64-64z" fill="#ffffff"/>'
    '<path d="M184 140c3 18 11 26 29 29-18 3-26 11-29 29-3-18-11-26-29-29 18-3 26-11 29-29z" fill="#ffffff" '
    'fill-opacity="0.9"/>'
    '</svg>')


def app_pixmap(size):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(APP_SVG.encode())).render(p)
    p.end()
    return pm


def app_icon():
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(app_pixmap(s))
    return ic


# ── 토글 스위치 ───────────────────────────────────────────────────────────

class Toggle(QCheckBox):
    """켜기/끄기 스위치. QCheckBox 와 사용법이 같다 (isChecked · setChecked · toggled)."""
    W, H = 38, 22

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _get_knob(self):
        return self._pos

    def _set_knob(self, v):
        self._pos = v
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def _animate(self, on):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def sizeHint(self):
        w = self.W
        if self.text():
            w += 10 + self.fontMetrics().horizontalAdvance(self.text())
        return QSize(w, max(self.H + 2, self.fontMetrics().height() + 4))

    def minimumSizeHint(self):
        return self.sizeHint()

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, _):
        # 애니메이션 중이 아니면 실제 상태로 그린다 (signals 를 막고 setChecked 한 경우 대비)
        pos = self._pos if self._anim.state() == QPropertyAnimation.State.Running else \
            (1.0 if self.isChecked() else 0.0)
        self._pos = pos
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.4)
        y = (self.height() - self.H) / 2
        off, on = QColor("#384054"), QColor(ACCENT)
        track = QColor(int(off.red() + (on.red() - off.red()) * pos), int(off.green() + (on.green() - off.green()) * pos),
                       int(off.blue() + (on.blue() - off.blue()) * pos))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, y, self.W, self.H), self.H / 2, self.H / 2)
        d = self.H - 6
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(3 + pos * (self.W - d - 6), y + 3, d, d))
        if self.text():
            p.setOpacity(1.0 if self.isEnabled() else 0.5)
            p.setPen(self.palette().color(QPalette.ColorRole.WindowText))
            p.drawText(QRectF(self.W + 10, 0, self.width() - self.W - 10, self.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


# ── 세그먼트 버튼 (몇 개 중 하나 고르기) ─────────────────────────────────────

class Segmented(QFrame):
    """[A | B | C] 중 하나를 고른다. QComboBox 와 같은 방식으로 쓴다
    (currentData · setCurrentIndex · count · itemData · currentIndexChanged)."""
    currentIndexChanged = Signal(int)

    def __init__(self, items, tip=""):
        super().__init__()
        self.setObjectName("segmented")
        self._data = [d for _, d in items]
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        for i, (label, _) in enumerate(items):
            b = QPushButton(label)
            b.setObjectName("segBtn")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._group.addButton(b, i)
            lay.addWidget(b)
        self._group.button(0).setChecked(True)
        self._index = 0
        self._group.idClicked.connect(self.setCurrentIndex)
        self.setToolTip(tip)

    def count(self):
        return len(self._data)

    def itemData(self, i):
        return self._data[i]

    def currentIndex(self):
        return self._index

    def currentData(self):
        return self._data[self._index]

    def setCurrentIndex(self, i):
        self._group.button(i).setChecked(True)
        if i != self._index:
            self._index = i
            self.currentIndexChanged.emit(i)


# ── 기능 카드: [아이콘 · 이름 · 설명 · 토글] + 켜면 보이는 세부 설정 ─────────────

class FeatureCard(QFrame):
    def __init__(self, icon_name, title, desc, toggle=None):
        super().__init__()
        self.setObjectName("card")
        self.toggle = toggle
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(12)
        badge = QLabel()
        badge.setObjectName("badge")
        badge.setFixedSize(34, 34)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(pixmap(icon_name, "#c9d0ff", 18))
        head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        self.desc = QLabel(desc)
        self.desc.setObjectName("muted")
        self.desc.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(self.desc)
        head.addLayout(col, 1)
        if toggle is not None:
            head.addWidget(toggle, 0, Qt.AlignmentFlag.AlignTop)
            toggle.toggled.connect(lambda _: self.sync())
        v.addLayout(head)
        self.body = QWidget()
        self.form = form_layout(self.body)
        v.addWidget(self.body)

    def sync(self):
        """토글이 꺼져 있으면 세부 설정을 접는다."""
        on = self.toggle is None or (self.toggle.isChecked() and self.toggle.isEnabled())
        self.body.setVisible(on and self.form.rowCount() > 0)


def form_layout(widget):
    f = QFormLayout(widget)
    f.setContentsMargins(0, 0, 0, 0)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    f.setHorizontalSpacing(14)
    f.setVerticalSpacing(10)
    return f


def button(text, icon_name=None, kind=None, tip=""):
    """kind: None(보통) | 'primary'(주요 동작) | 'danger'(중지) | 'ghost'(테두리 없음)."""
    b = QPushButton(text)
    if icon_name:
        b.setIcon(icon(icon_name, "#ffffff" if kind == "primary" else DANGER if kind == "danger" else TEXT))
        b.setIconSize(QSize(16, 16))
    if kind:
        b.setObjectName(kind)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    return b


# ── 끌어다 놓기 목록: 비어 있으면 안내를 그린다 ─────────────────────────────

class DropList(QListWidget):
    def __init__(self, hint):
        super().__init__()
        self.setObjectName("drop")
        self.hint = hint
        # 긴 파일 이름은 가운데를 줄여 보여 준다 (가로 스크롤 없이)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        m = self.model()
        m.rowsInserted.connect(self._sync)
        m.rowsRemoved.connect(self._sync)
        m.modelReset.connect(self._sync)
        self._sync()

    def _sync(self, *_):
        self.setProperty("empty", self.count() == 0)
        self.style().unpolish(self)
        self.style().polish(self)
        self.viewport().update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.count():
            return
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.viewport().rect()
        p.drawPixmap(QPointF(r.center().x() - 13, r.center().y() - 30), pixmap("upload", MUTED, 26))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(r.left(), r.center().y() + 2, r.width(), 40),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.hint)


def note(text):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setObjectName("note")
    return lbl


# ── 테마 ──────────────────────────────────────────────────────────────────

def _arrow_files():
    """콤보·숫자칸의 화살표 그림 (스타일시트는 파일 경로로만 그림을 받는다)."""
    d = CACHE / "ui"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("chevron-down", "chevron-up"):
        f = d / f"{name}.svg"
        f.write_text(_svg(name, MUTED, 2.0), encoding="utf-8")
        out[name] = f.as_posix()
    return out


def apply_theme(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, FIELD), (QPalette.ColorRole.AlternateBase, SURFACE),
        (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, RAISED),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, RAISED),
        (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, FAINT),
    ):
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.ButtonText, QPalette.ColorRole.Text, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(FAINT))
    app.setPalette(pal)
    a = _arrow_files()
    app.setStyleSheet(f"""
QWidget {{ color:{TEXT}; }}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background:{BG}; }}
QToolTip {{ background:{RAISED}; color:{TEXT}; border:1px solid {BORDER}; border-radius:6px; padding:6px 8px; }}

/* 섹션 */
QGroupBox {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; margin-top:28px;
             padding:14px 12px 12px 12px; }}
QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left; left:6px; top:2px; padding:0 2px;
                    color:{TEXT}; font-weight:600; font-size:10pt; }}
QFrame#card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; }}
QLabel#badge {{ background:#262c45; border-radius:10px; }}
QLabel#cardTitle {{ font-weight:600; font-size:10pt; }}
QLabel#muted, QLabel#note {{ color:{MUTED}; }}
QLabel#note {{ font-size:8.5pt; }}

/* 탭: 설정 탭은 알약 모양 */
QTabWidget::pane {{ border:none; background:transparent; top:-2px; }}
QTabBar::tab {{ background:transparent; color:{MUTED}; padding:6px 8px; margin:0 2px 8px 0; border-radius:8px; }}
QTabBar::tab:hover {{ color:{TEXT}; background:#1d2230; }}
QTabBar::tab:selected {{ background:{RAISED}; color:{TEXT}; font-weight:600; }}
QTabWidget#mainTabs::pane {{ border:none; border-top:1px solid {BORDER}; top:0; }}
QTabWidget#mainTabs > QTabBar::tab {{ padding:9px 20px; margin:10px 6px 10px 0; font-size:10pt; font-weight:600;
                                       border-radius:10px; }}
QTabWidget#mainTabs > QTabBar::tab:selected {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {ACCENT},
                                                stop:1 {ACCENT2}); color:#ffffff; }}

/* 버튼 */
QPushButton {{ background:{RAISED}; border:1px solid {BORDER}; border-radius:9px; padding:6px 12px; min-height:20px; }}
QPushButton:hover {{ background:#2a3142; border-color:{BORDER_HI}; }}
QPushButton:pressed {{ background:#1d2230; }}
QPushButton:disabled {{ background:{SURFACE}; color:{FAINT}; border-color:#20242f; }}
QPushButton#primary {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {ACCENT},stop:1 {ACCENT2});
                       border:none; color:#ffffff; font-weight:600; padding:9px 24px; border-radius:11px; }}
QPushButton#primary:hover {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7d8bff,stop:1 #aa80ff); }}
QPushButton#primary:disabled {{ background:#252a37; color:{FAINT}; }}
QPushButton#danger {{ background:transparent; border:1px solid #4a2c35; color:{DANGER}; border-radius:11px; padding:9px 16px; }}
QPushButton#danger:hover {{ background:#2a1a20; }}
QPushButton#danger:disabled {{ border-color:#22252f; color:{FAINT}; }}
QPushButton#ghost {{ background:transparent; border:none; color:{MUTED}; padding:4px 8px; }}
QPushButton#ghost:hover {{ color:{TEXT}; }}

/* 세그먼트 */
QFrame#segmented {{ background:{FIELD}; border:1px solid {BORDER}; border-radius:10px; }}
QPushButton#segBtn {{ background:transparent; border:none; border-radius:7px; padding:5px 10px; color:{MUTED}; }}
QPushButton#segBtn:hover {{ color:{TEXT}; }}
QPushButton#segBtn:checked {{ background:{RAISED}; color:{TEXT}; font-weight:600; border:1px solid {BORDER_HI}; }}

/* 입력칸 */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background:{FIELD}; border:1px solid {BORDER}; border-radius:8px;
    padding:5px 10px; min-height:20px; selection-background-color:{ACCENT}; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color:{BORDER_HI}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color:{ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{ color:{FAINT}; background:{SURFACE}; }}
QComboBox::drop-down {{ border:none; width:26px; }}
QComboBox::down-arrow {{ image:url("{a['chevron-down']}"); width:14px; height:14px; }}
QComboBox QAbstractItemView {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:4px; outline:0;
                               selection-background-color:#2b3352; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border:none; width:20px; background:transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image:url("{a['chevron-up']}"); width:12px; height:12px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image:url("{a['chevron-down']}"); width:12px; height:12px; }}

/* 슬라이더 */
QSlider::groove:horizontal {{ height:4px; background:#2a3040; border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {ACCENT},stop:1 {ACCENT2});
                                border-radius:2px; }}
QSlider::handle:horizontal {{ background:#ffffff; width:14px; height:14px; margin:-5px 0; border-radius:7px; }}
QSlider::handle:horizontal:hover {{ background:#e6e9ff; }}
QSlider::sub-page:horizontal:disabled {{ background:#3a4050; }}
QSlider::handle:horizontal:disabled {{ background:#5d6577; }}

/* 진행 막대 */
QProgressBar {{ background:{FIELD}; border:1px solid {BORDER}; border-radius:8px; min-height:16px; text-align:center;
                color:{TEXT}; font-size:8.5pt; }}
QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {ACCENT},stop:1 {ACCENT2});
                       border-radius:7px; }}

/* 목록 */
QListWidget {{ background:{FIELD}; border:1px solid {BORDER}; border-radius:10px; padding:4px; outline:0; }}
QListWidget#drop[empty="true"] {{ border:1.5px dashed #3a4258; }}
QListWidget::item {{ padding:6px 8px; border-radius:7px; }}
QListWidget::item:hover {{ background:#1a1f2b; }}
QListWidget::item:selected {{ background:#28304c; color:{TEXT}; }}

/* 스크롤바 */
QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
QScrollBar::handle:vertical {{ background:#2c3242; border-radius:4px; min-height:32px; }}
QScrollBar::handle:vertical:hover {{ background:{BORDER_HI}; }}
QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:#2c3242; border-radius:4px; min-width:32px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}

QSplitter::handle {{ background:transparent; width:8px; }}
QStatusBar {{ background:{BG}; color:{MUTED}; }}
QStatusBar::item {{ border:none; }}
QMenu {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:4px; }}
QMenu::item:selected {{ background:#2b3352; border-radius:6px; }}
""")
