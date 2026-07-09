"""
视频输入和窗口适配工具。
摄像头、视频文件、RTSP 流都从这里打开；YOLO 输出也在这里转成跟踪器需要的数组。
显示窗口被拉大/缩小时，这里负责等比例放画面，避免文字和检测框被直接拉变形。
"""

from __future__ import annotations

import cv2
import numpy as np


WINDOW_NAME = "VisionGuard AI"


def open_source(source: str) -> cv2.VideoCapture:
    """打开摄像头编号、本地视频或网络视频流。"""
    source = str(source)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def detections_from_result(result) -> np.ndarray:
    """把 Ultralytics 检测结果转成 [x1, y1, x2, y2, score]。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf], axis=1).astype(np.float32)


def resolve_display_size(frame_width: int, frame_height: int, display_width: int, display_height: int) -> tuple[int, int]:
    """计算预览窗口大小；宽或高写 0 时按原比例自动补齐。"""
    if display_width <= 0 and display_height <= 0:
        return max(1, frame_width), max(1, frame_height)
    if display_width <= 0:
        scale = display_height / max(frame_height, 1)
        display_width = int(frame_width * scale)
    elif display_height <= 0:
        scale = display_width / max(frame_width, 1)
        display_height = int(frame_height * scale)
    return max(1, int(display_width)), max(1, int(display_height))


def get_current_window_size(default_width: int, default_height: int) -> tuple[int, int]:
    """读取当前窗口大小，读不到就用默认尺寸。"""
    if not hasattr(cv2, "getWindowImageRect"):
        return default_width, default_height
    try:
        _, _, window_width, window_height = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return default_width, default_height
    if window_width <= 1 or window_height <= 1:
        return default_width, default_height
    return int(window_width), int(window_height)


def make_display_canvas(frame: np.ndarray, canvas_width: int, canvas_height: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """先做等比例 letterbox，再绘制 UI，避免窗口自适应时文字变形。"""
    frame_height, frame_width = frame.shape[:2]
    canvas_width = max(1, int(canvas_width))
    canvas_height = max(1, int(canvas_height))
    scale = min(canvas_width / max(frame_width, 1), canvas_height / max(frame_height, 1))
    resized_width = max(1, int(round(frame_width * scale)))
    resized_height = max(1, int(round(frame_height * scale)))
    offset_x = (canvas_width - resized_width) // 2
    offset_y = (canvas_height - resized_height) // 2

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=frame.dtype)
    canvas[:] = (12, 14, 18)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas, scale, (offset_x, offset_y)


def transform_tracks_for_display(tracks: np.ndarray, scale: float, offset: tuple[int, int]) -> np.ndarray:
    """把原始画面坐标里的跟踪框映射到显示画布坐标。"""
    if tracks.size == 0:
        return tracks
    offset_x, offset_y = offset
    mapped = tracks.copy()
    mapped[:, [0, 2]] = mapped[:, [0, 2]] * scale + offset_x
    mapped[:, [1, 3]] = mapped[:, [1, 3]] * scale + offset_y
    return mapped


def transform_points_for_display(
    points: list[tuple[float, float]] | None,
    scale: float,
    offset: tuple[int, int],
) -> list[tuple[float, float]] | None:
    """把 ROI 点从原始画面坐标映射到显示画布坐标。"""
    if points is None:
        return None
    offset_x, offset_y = offset
    return [(x * scale + offset_x, y * scale + offset_y) for x, y in points]


def transform_line_for_display(
    line: tuple[tuple[float, float], tuple[float, float]] | None,
    scale: float,
    offset: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """把警戒线从原始画面坐标映射到显示画布坐标。"""
    if line is None:
        return None
    mapped = transform_points_for_display([line[0], line[1]], scale, offset)
    return mapped[0], mapped[1]
