"""
几何计算
规则模块经常要用“脚底中心点”、点是否在多边形里、两点距离等，
"""

from __future__ import annotations

from math import hypot
from typing import Sequence


Point = tuple[float, float]
BBox = Sequence[float]


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox[:4]
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def bbox_bottom_center(bbox: BBox) -> Point:
    x1, _, x2, y2 = bbox[:4]
    return (float(x1 + x2) * 0.5, float(y2))


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(point: Point, polygon: Sequence[Point] | None) -> bool:
    if not polygon:
        return True
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside
