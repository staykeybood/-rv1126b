"""
画面叠加层。
ROI、警戒线、跟踪框、左上角状态栏和右上角报警栏。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


EVENT_LABELS_CN = {
    "intrusion": "区域入侵",
    "loitering": "异常停留",
    "wandering": "徘徊",
    "repeat_approach": "多次靠近敏感区域",
    "line_crossing": "穿越警戒线",
    "after_hours_intrusion": "非工作时间人员出现",
    "camera_block": "遮挡/贴近摄像头",
    "running": "快速奔跑",
}
EVENT_SHORT_LABELS_CN = {
    "intrusion": "区域入侵",
    "loitering": "异常停留",
    "wandering": "徘徊",
    "repeat_approach": "多次靠近",
    "line_crossing": "越线",
    "after_hours_intrusion": "非工作时间",
    "camera_block": "镜头遮挡",
    "running": "快速奔跑",
}
LEVEL_LABELS_CN = {"critical": "严重", "warning": "预警", "info": "提示"}
LEVEL_COLORS = {"critical": (30, 55, 245), "warning": (0, 170, 255), "info": (255, 200, 80)}
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_FONT_CACHE: dict[int, object | None] = {}


def update_recent_events(recent_events: list[dict], new_events: list[dict], max_events: int = 8) -> None:
    """同一个 ID 的同一种报警只保留最新一条，避免报警栏被重复信息刷满。"""
    for event in new_events:
        key = (event.get("type"), event.get("track_id"))
        recent_events[:] = [
            old
            for old in recent_events
            if (old.get("type"), old.get("track_id")) != key
        ]
        recent_events.append(event)
    del recent_events[:-max_events]


def load_ui_font(size: int):
    """优先加载中文字体；找不到字体时后面会退回英文显示。"""
    if ImageFont is None:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size)
        except OSError:
            continue
        _FONT_CACHE[size] = font
        return font
    _FONT_CACHE[size] = None
    return None


def blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], alpha: float) -> None:
    """画半透明底色，用来让文字在复杂背景上更清楚。"""
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return
    roi = frame[y1:y2, x1:x2]
    tint = np.empty_like(roi)
    tint[:] = color
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, dst=roi)


def add_text(
    text_items: list[dict],
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 18,
    scale: float = 0.55,
    thickness: int = 1,
    fallback: str | None = None,
) -> None:
    """先把文字排队，最后统一绘制，减少 Pillow/OpenCV 来回转换。"""
    text_items.append(
        {
            "text": text,
            "xy": xy,
            "color": color,
            "size": size,
            "scale": scale,
            "thickness": thickness,
            "fallback": fallback or text,
        }
    )


def render_texts(frame: np.ndarray, text_items: list[dict], chinese_ui: bool = True) -> None:
    """绘制文字：中文走 Pillow，板端没有字体时自动用 OpenCV 英文兜底。"""
    if not text_items:
        return
    can_draw_chinese = chinese_ui and Image is not None and ImageDraw is not None and load_ui_font(18) is not None
    if can_draw_chinese:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(pil_image)
        for item in text_items:
            font = load_ui_font(int(item["size"]))
            b, g, r = item["color"]
            drawer.text(item["xy"], item["text"], font=font, fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        return

    for item in text_items:
        text = item["fallback"].encode("ascii", "ignore").decode("ascii").strip()
        if not text:
            text = "event"
        x, y = item["xy"]
        cv2.putText(
            frame,
            text,
            (x, y + int(item["size"] * 0.78)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(item["scale"]),
            item["color"],
            int(item["thickness"]),
            cv2.LINE_AA,
        )


def event_label_cn(event: dict) -> str:
    """拿报警类型的完整中文名。"""
    return str(event.get("type_cn") or EVENT_LABELS_CN.get(event.get("type"), event.get("type", "事件")))


def event_short_label_cn(event: dict) -> str:
    """拿报警栏里更短的中文名，避免一行太长。"""
    return EVENT_SHORT_LABELS_CN.get(str(event.get("type")), event_label_cn(event))


def event_level_color(level: str | None) -> tuple[int, int, int]:
    """按报警等级选颜色：严重偏红，预警偏黄，提示偏蓝。"""
    return LEVEL_COLORS.get(str(level or "info"), LEVEL_COLORS["info"])


def ui_metrics(frame_shape: tuple[int, int]) -> dict[str, float | int]:
    """根据当前画布大小算一套 UI 尺寸。

    摄像头窗口被拉伸后，画布尺寸会变；文字、面板、线宽都跟着这个比例走。
    """
    height, width = frame_shape
    base = max(1, min(width, height))
    scale = float(np.clip(base / 720.0, 0.70, 1.35))
    return {
        "scale": scale,
        "margin": int(max(6, round(10 * scale))),
        "pad": int(max(8, round(12 * scale))),
        "title_size": int(max(13, round(18 * scale))),
        "body_size": int(max(11, round(15 * scale))),
        "small_size": int(max(10, round(13 * scale))),
        "row_h": int(max(24, round(32 * scale))),
        "header_h": int(max(25, round(34 * scale))),
        "line_thick": int(max(1, round(2 * scale))),
        "box_thick": int(max(1, round(2 * scale))),
    }


def estimate_text_width(text: str, size: int) -> int:
    """粗略估算文字宽度，够面板自适应用，不追求像素级精确。"""
    width = 0.0
    for char in text:
        width += size * (0.55 if ord(char) < 128 else 0.95)
    return int(width)


def fit_text(text: str, max_width: int, size: int) -> str:
    """文本太长时从尾部省略，避免挤出面板。"""
    if estimate_text_width(text, size) <= max_width:
        return text
    suffix = "..."
    text = text.strip()
    while text and estimate_text_width(text + suffix, size) > max_width:
        text = text[:-1]
    return (text + suffix) if text else suffix


def rect_area(rect: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = rect
    return max(0, x2 - x1) * max(0, y2 - y1)


def rect_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return rect_area((x1, y1, x2, y2))


def choose_panel_rect(
    frame_shape: tuple[int, int],
    panel_size: tuple[int, int],
    avoid_rects: list[tuple[int, int, int, int]],
    reserved_rects: list[tuple[int, int, int, int]] | None = None,
    prefer: tuple[str, ...] = ("top_left", "top_right", "bottom_right", "bottom_left"),
) -> tuple[int, int, int, int]:
    """给信息面板选一个尽量不挡 ROI 的角落位置。"""
    height, width = frame_shape
    panel_w, panel_h = panel_size
    margin = max(8, min(width, height) // 60)
    panel_w = min(panel_w, max(1, width - 2 * margin))
    panel_h = min(panel_h, max(1, height - 2 * margin))
    candidates = {
        "top_left": (margin, margin, margin + panel_w, margin + panel_h),
        "top_right": (width - panel_w - margin, margin, width - margin, margin + panel_h),
        "bottom_right": (width - panel_w - margin, height - panel_h - margin, width - margin, height - margin),
        "bottom_left": (margin, height - panel_h - margin, margin + panel_w, height - margin),
    }
    reserved_rects = reserved_rects or []
    weighted_avoid = [(rect, 1.0) for rect in avoid_rects] + [(rect, 4.0) for rect in reserved_rects]
    best_rect = candidates[prefer[0]]
    best_score = float("inf")
    for order, key in enumerate(prefer):
        rect = candidates[key]
        score = sum(rect_intersection_area(rect, avoid) * weight for avoid, weight in weighted_avoid)
        score += order * 0.01
        if score < best_score:
            best_score = score
            best_rect = rect
    return best_rect


def draw_corner_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """画角标风格的人体跟踪框，比普通矩形框轻一点。"""
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    corner = int(max(14, min(34, min(box_w, box_h) * 0.28)))
    shadow = (8, 12, 16)

    for offset, line_color, line_thick in ((2, shadow, thickness + 2), (0, color, thickness)):
        cv2.line(frame, (x1, y1 + offset), (x1 + corner, y1 + offset), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x1 + offset, y1), (x1 + offset, y1 + corner), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x2 - corner, y1 + offset), (x2, y1 + offset), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x2 - offset, y1), (x2 - offset, y1 + corner), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y2 - offset), (x1 + corner, y2 - offset), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x1 + offset, y2 - corner), (x1 + offset, y2), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x2 - corner, y2 - offset), (x2, y2 - offset), line_color, line_thick, cv2.LINE_AA)
        cv2.line(frame, (x2 - offset, y2 - corner), (x2 - offset, y2), line_color, line_thick, cv2.LINE_AA)


def draw_status_panel(
    frame: np.ndarray,
    text_items: list[dict],
    counts: dict,
    runtime_fps: float | None,
    process_every: int,
    roi_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """画左侧状态栏：人数、禁区人数、FPS 和检测间隔。"""
    height, width = frame.shape[:2]
    metrics = ui_metrics(frame.shape[:2])
    pad = int(metrics["pad"])
    title_size = int(metrics["title_size"])
    body_size = int(metrics["body_size"])
    small_size = int(metrics["small_size"])
    line_thick = int(metrics["line_thick"])

    title = "智安视觉监测系统"
    title_fallback = "VisionGuard AI"
    if runtime_fps is None:
        compact_status = f"人数 {counts['persons']}  禁区 {counts['roi_persons']}"
        fallback_status = f"Persons {counts['persons']}  ROI {counts['roi_persons']}"
        status_rows = [(compact_status, fallback_status)]
    else:
        compact_status = f"人数 {counts['persons']}  禁区 {counts['roi_persons']}  FPS {runtime_fps:.1f}  间隔 {process_every}"
        fallback_status = f"Persons {counts['persons']}  ROI {counts['roi_persons']}  FPS {runtime_fps:.1f}  Int {process_every}"
        short_row = f"人数 {counts['persons']}  禁区 {counts['roi_persons']}"
        fps_row = f"FPS {runtime_fps:.1f}  间隔 {process_every}"
        fallback_short = f"Persons {counts['persons']}  ROI {counts['roi_persons']}"
        fallback_fps = f"FPS {runtime_fps:.1f}  Int {process_every}"
        status_rows = [(compact_status, fallback_status)]

    min_panel_w = int(max(210 * metrics["scale"], estimate_text_width(title, title_size) + 2 * pad))
    max_panel_w = max(1, width - 2 * int(metrics["margin"]))
    panel_w = min(max(min_panel_w, int(width * 0.34)), max_panel_w)
    content_w = panel_w - 2 * pad
    if runtime_fps is not None and estimate_text_width(compact_status, body_size) > content_w:
        status_rows = [(short_row, fallback_short), (fps_row, fallback_fps)]

    row_gap = max(2, int(4 * metrics["scale"]))
    panel_h = pad + title_size + row_gap + len(status_rows) * (body_size + row_gap) + pad // 2
    x1, y1, x2, y2 = choose_panel_rect(
        frame.shape[:2],
        (panel_w, panel_h),
        roi_rects,
        prefer=("top_left", "bottom_left"),
    )
    blend_rect(frame, x1, y1, x2, y2, (18, 24, 30), 0.66)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (74, 104, 116), 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y1), (80, 220, 120), line_thick, cv2.LINE_AA)

    add_text(text_items, title, (x1 + pad, y1 + pad - 2), (220, 245, 245), title_size, 0.5, 2, title_fallback)
    text_y = y1 + pad + title_size + row_gap
    for row, fallback in status_rows:
        row = fit_text(row, content_w, body_size)
        add_text(text_items, row, (x1 + pad, text_y), (90, 235, 245), body_size, 0.45, 2, fallback)
        text_y += body_size + row_gap
    return x1, y1, x2, y2


def draw_alarm_panel(
    frame: np.ndarray,
    text_items: list[dict],
    events: list[dict],
    max_rows: int = 3,
) -> tuple[int, int, int, int] | None:
    """画右上角报警栏，只显示最近几条，避免遮住太多画面。"""
    if not events or max_rows <= 0:
        return None
    height, width = frame.shape[:2]
    metrics = ui_metrics(frame.shape[:2])
    margin = int(metrics["margin"])
    pad = int(metrics["pad"])
    title_size = int(metrics["title_size"])
    body_size = int(metrics["body_size"])
    row_h = int(metrics["row_h"])
    header_h = int(metrics["header_h"])
    line_thick = int(metrics["line_thick"])

    shown_events = list(reversed(events[-max_rows:]))
    min_panel_w = int(max(230 * metrics["scale"], estimate_text_width("报警中心 3", title_size) + 2 * pad))
    panel_w = min(max(min_panel_w, int(width * 0.34)), max(1, width - 2 * margin))
    panel_h = header_h + row_h * len(shown_events) + pad
    x1 = max(margin, width - panel_w - margin)
    y1 = margin
    x2 = min(width - margin, x1 + panel_w)
    y2 = min(height - margin, y1 + panel_h)
    content_w = max(1, x2 - x1 - 2 * pad)

    blend_rect(frame, x1, y1, x2, y2, (18, 20, 26), 0.68)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (78, 82, 96), 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y1), (0, 170, 255), line_thick, cv2.LINE_AA)
    add_text(text_items, f"报警中心 {len(shown_events)}", (x1 + pad, y1 + pad - 2), (235, 240, 245), title_size, 0.5, 2, "ALERTS")

    for idx, event in enumerate(shown_events):
        row_y = y1 + header_h + idx * row_h
        row_bottom = min(y2 - margin // 2, row_y + row_h - max(3, margin // 3))
        level = str(event.get("level", "info"))
        color = event_level_color(level)
        blend_rect(frame, x1 + pad // 2, row_y, x2 - pad // 2, row_bottom, (25, 29, 36), 0.68)
        cv2.rectangle(frame, (x1 + pad // 2, row_y), (x1 + pad // 2 + max(3, line_thick * 2), row_bottom), color, -1)
        level_cn = LEVEL_LABELS_CN.get(level, "提示")
        tid = event.get("track_id", "-")
        type_cn = event_short_label_cn(event)
        title = fit_text(f"{level_cn} {type_cn} ID {tid}", content_w - pad, body_size)
        fallback = f"{level.upper()} {event.get('type', 'event')} ID {tid}"
        add_text(text_items, title, (x1 + pad + 5, row_y + max(4, (row_h - body_size) // 2)), color, body_size, 0.45, 2, fallback)
    return x1, y1, x2, y2


def draw(
    frame: np.ndarray,
    tracks: np.ndarray,
    counts: dict,
    events: list[dict],
    roi,
    line=None,
    runtime_fps: float | None = None,
    process_every: int = 1,
    chinese_ui: bool = True,
    max_alert_rows: int = 3,
) -> None:
    """总绘制入口：先画区域和框，再画状态栏/报警栏，最后统一画文字。"""
    text_items: list[dict] = []
    roi_rects: list[tuple[int, int, int, int]] = []
    metrics = ui_metrics(frame.shape[:2])
    line_thick = int(metrics["line_thick"])
    box_thick = int(metrics["box_thick"])
    small_size = int(metrics["small_size"])
    label_pad = max(5, int(7 * metrics["scale"]))
    label_h = max(20, int(small_size + 10 * metrics["scale"]))

    if roi:
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        roi_points = pts.reshape(-1, 2)
        rx1, ry1 = roi_points.min(axis=0)
        rx2, ry2 = roi_points.max(axis=0)
        roi_rects.append((int(rx1), int(ry1), int(rx2), int(ry2)))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 120, 200))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)
        cv2.polylines(frame, [pts], True, (0, 210, 255), line_thick, cv2.LINE_AA)
        for point in roi_points:
            cv2.circle(frame, tuple(point), max(2, line_thick + 1), (0, 210, 255), -1, cv2.LINE_AA)
    if line:
        p1, p2 = line
        point1 = (int(p1[0]), int(p1[1]))
        point2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame, point1, point2, (30, 55, 245), max(2, line_thick + 1), cv2.LINE_AA)
        cv2.circle(frame, point1, max(3, line_thick + 2), (30, 55, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, point2, max(3, line_thick + 2), (30, 55, 245), -1, cv2.LINE_AA)

    alerted_tracks = {}
    for event in events:
        track_id = event.get("track_id")
        if track_id is not None:
            alerted_tracks[int(track_id)] = str(event.get("level", "info"))
    for track in tracks:
        x1, y1, x2, y2, track_id, score = track
        tid = int(track_id)
        color = event_level_color(alerted_tracks.get(tid)) if tid in alerted_tracks else (80, 220, 120)
        bbox = (int(x1), int(y1), int(x2), int(y2))
        draw_corner_box(frame, bbox, color, box_thick)
        label = f"ID {int(track_id)} {score:.2f}"
        label_w = estimate_text_width(label, small_size) + 2 * label_pad
        lx = max(0, int(x1))
        ly = max(0, int(y1) - label_h - 2)
        if ly <= 2:
            ly = min(frame.shape[0] - label_h - 2, int(y1) + 4)
        blend_rect(frame, lx, ly, lx + label_w, ly + label_h, (12, 16, 20), 0.78)
        cv2.rectangle(frame, (lx, ly), (lx + label_w, ly + label_h), color, 1, cv2.LINE_AA)
        add_text(text_items, label, (lx + label_pad, ly + max(3, (label_h - small_size) // 2)), color, small_size, 0.5, 2, label)

    draw_status_panel(frame, text_items, counts, runtime_fps, process_every, roi_rects)
    draw_alarm_panel(frame, text_items, events, max_alert_rows)
    render_texts(frame, text_items, chinese_ui=chinese_ui)
