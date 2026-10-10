"""Dark palette shared with the Voxprint suite. Text colours stay opaque."""

from __future__ import annotations

TEXT = "#f2f2f5"
TEXT_MUTED = "#c4c4d0"
TEXT_FAINT = "#a8a8b8"
TEXT_DISABLED = "#8e8e9e"
TEXT_ON_ACCENT = "#0b1220"
ACCENT = "#60a5fa"
ACCENT_HOVER = "#7db9ff"
ACCENT_SOFT = "#93c5fd"
ACCENT_STRONG = "#2563eb"
ROOT_PLAIN = "#17171c"
ROOT_GLASS = "rgba(16,16,22,222)"
CARD_PLAIN = "#202029"
CARD_GLASS = "rgba(34,34,44,238)"
CONTROL_BG = "#2c2c38"
CONTROL_HOVER = "#383846"
CONTROL_BORDER = "#4b4b5c"
AMBER = "#fcd34d"
NOTE_BG = "#2a2418"
NOTE_BORDER = "#6b5720"
GREEN = "#86efac"
BADGE_GREEN_TEXT = "#0b1f12"
BADGE_AMBER_TEXT = "#251a02"
PRIMARY = "#f5d76e"
PRIMARY_HOVER = "#f9e291"
PRIMARY_BORDER = "#fbe8a6"
TEXT_ON_PRIMARY = "#1c1604"
PRIMARY_DISABLED_BG = "#36321f"
PRIMARY_DISABLED_TEXT = "#b9b08a"
MENU_BG = "#1c2535"
MENU_SELECTION = "#2f4c78"
ROOT_GLASS_MORE = "rgba(16,16,22,167)"
CARD_GLASS_MORE = "rgba(34,34,44,214)"
FAILED_TEXT = "#2a0a0a"
FAILED_BG = "#fca5a5"

CONTRAST_PAIRS = [
    (TEXT, ROOT_PLAIN),
    (TEXT, CARD_PLAIN),
    (TEXT, CONTROL_BG),
    (TEXT, CONTROL_HOVER),
    (TEXT_MUTED, ROOT_PLAIN),
    (TEXT_MUTED, CARD_PLAIN),
    (TEXT_FAINT, ROOT_PLAIN),
    (TEXT_FAINT, CARD_PLAIN),
    (TEXT_ON_ACCENT, ACCENT),
    (TEXT_ON_ACCENT, ACCENT_HOVER),
    (TEXT_ON_ACCENT, ACCENT_SOFT),
    (TEXT, ACCENT_STRONG),
    ("#dcdce4", "#34343f"),
    ("#86efac", CARD_PLAIN),
    (BADGE_GREEN_TEXT, GREEN),
    (BADGE_AMBER_TEXT, AMBER),
    (AMBER, CARD_PLAIN),
    (AMBER, NOTE_BG),
    (AMBER, ROOT_PLAIN),
    ("#ffffff", ACCENT_STRONG),
    (ACCENT_SOFT, ROOT_PLAIN),
    (ACCENT_SOFT, CARD_PLAIN),
    (TEXT_ON_PRIMARY, PRIMARY),
    (TEXT_ON_PRIMARY, PRIMARY_HOVER),
    (PRIMARY_DISABLED_TEXT, PRIMARY_DISABLED_BG),
    (TEXT, MENU_BG),
    ("#ffffff", MENU_SELECTION),
    (FAILED_TEXT, FAILED_BG),
]

_STYLE = """
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans", sans-serif;
    font-size: 14px; color: {text}; }}
QWidget#root {{ background: __ROOT_BG__; color: {text}; }}
QLabel {{ color: {text}; background: transparent; }}
QLabel#title {{ font-size: 22px; font-weight: 600; color: {text}; }}
QLabel#subtitle {{ color: {muted}; }}
QLabel#footer {{ color: {faint}; font-size: 12px; }}
QFrame#card {{ background: __CARD_BG__; border: 1px solid {border}; border-radius: 12px; }}
QPushButton {{ background: {control}; color: {text}; border: 1px solid {border}; border-radius: 8px;
    padding: 6px 14px; }}
QPushButton:hover {{ background: {hover}; }}
QPushButton:disabled {{ color: {disabled}; background: #1f1f28; border-color: #34343f; }}
QPushButton#primary {{ background: {primary}; border: 1px solid {primary_border}; font-weight: 600;
    color: {on_primary}; }}
QPushButton#primary:hover {{ background: {primary_hover}; }}
QPushButton#primary:disabled {{ background: {primary_dis_bg}; color: {primary_dis_text}; border-color: #4a4430; }}
QLineEdit, QPlainTextEdit {{ background: {control}; color: {text}; border: 1px solid {border}; border-radius: 8px;
    padding: 6px 10px; selection-background-color: {strong}; selection-color: #ffffff; }}
QTableWidget {{ background: __CARD_BG__; color: {text}; border: 1px solid {border}; border-radius: 8px;
    gridline-color: {border}; }}
QTableWidget::item:selected {{ background: {menu_sel}; color: #ffffff; }}
QHeaderView::section {{ background: {control}; color: {text}; border: none; padding: 6px; }}
QProgressBar {{ background: {control}; color: {text}; border: 1px solid {border}; border-radius: 8px;
    text-align: center; }}
QProgressBar::chunk {{ background: {strong}; border-radius: 7px; }}
QCheckBox {{ color: {text}; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {accent}; border-radius: 4px;
    background: {control}; }}
QCheckBox::indicator:checked {{ background: {strong}; border-color: {soft}; }}
QComboBox {{ background: {control}; color: {text}; border: 1px solid {border}; border-radius: 8px; padding: 6px 12px; }}
QComboBox QAbstractItemView {{ background: {menu_bg}; color: {text}; selection-background-color: {menu_sel};
    selection-color: #ffffff; }}
QMenu {{ background: {menu_bg}; color: {text}; border: 1px solid {border}; }}
QMenu::item:selected {{ background: {menu_sel}; color: #ffffff; }}
QLabel#badge[state="failed"] {{ color: {failed_text}; background: {failed_bg}; padding: 2px 8px; border-radius: 8px;
    font-weight: 700; }}
""".format(
    text=TEXT,
    muted=TEXT_MUTED,
    faint=TEXT_FAINT,
    disabled=TEXT_DISABLED,
    accent=ACCENT,
    soft=ACCENT_SOFT,
    strong=ACCENT_STRONG,
    control=CONTROL_BG,
    hover=CONTROL_HOVER,
    border=CONTROL_BORDER,
    primary=PRIMARY,
    primary_hover=PRIMARY_HOVER,
    primary_border=PRIMARY_BORDER,
    on_primary=TEXT_ON_PRIMARY,
    primary_dis_bg=PRIMARY_DISABLED_BG,
    primary_dis_text=PRIMARY_DISABLED_TEXT,
    menu_bg=MENU_BG,
    menu_sel=MENU_SELECTION,
    failed_text=FAILED_TEXT,
    failed_bg=FAILED_BG,
)


def build_style(glass: bool) -> str:
    """Stylesheet. Glass tints only the window and card fills; text stays the same opaque colours."""
    root_bg = ROOT_GLASS if glass else ROOT_PLAIN
    card_bg = CARD_GLASS if glass else CARD_PLAIN
    return _STYLE.replace("__ROOT_BG__", root_bg).replace("__CARD_BG__", card_bg)


def _channel(value: str) -> float:
    channel = int(value, 16) / 255
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    text = color.removeprefix("#")
    red = _channel(text[0:2])
    green = _channel(text[2:4])
    blue = _channel(text[4:6])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(relative_luminance(foreground), relative_luminance(background))
    darker = min(relative_luminance(foreground), relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)
