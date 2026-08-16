"""matplotlib の日本語フォント設定。

AQWA の 04_Visualization/font_utils.py と同じ優先順位で、環境にあるフォントを選ぶ。
"""

from __future__ import annotations

import os
import platform

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_MAC_FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
]

_MAC_FONTS = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Kaku Gothic Pro",
    "Yu Gothic Medium",
    "Yu Gothic",
]

_LINUX_FONTS = [
    "IPAexGothic",
    "IPAGothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "VL PGothic",
]

_WINDOWS_FONTS = ["Yu Gothic UI", "Yu Gothic", "Meiryo UI", "Meiryo", "MS Gothic"]

_FALLBACK = ["DejaVu Sans", "Arial", "sans-serif"]


def _add_mac_fonts() -> int:
    added = 0
    for font_dir in _MAC_FONT_DIRS:
        if not os.path.isdir(font_dir):
            continue
        for name in os.listdir(font_dir):
            if not name.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            try:
                fm.fontManager.addfont(os.path.join(font_dir, name))
                added += 1
            except Exception:  # noqa: BLE001 - 壊れたフォントは黙って飛ばす
                pass
    return added


def setup_japanese_font() -> list[str]:
    system = platform.system()
    if system == "Darwin":
        _add_mac_fonts()
        candidates = _MAC_FONTS
    elif system == "Linux":
        candidates = _MAC_FONTS if _add_mac_fonts() else _LINUX_FONTS
    elif system == "Windows":
        candidates = _WINDOWS_FONTS
    else:
        candidates = []

    available = {f.name for f in fm.fontManager.ttflist}
    fonts = [f for f in [*candidates, *_FALLBACK] if f in available or f == "sans-serif"]
    if not fonts:
        fonts = ["sans-serif"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    return fonts
