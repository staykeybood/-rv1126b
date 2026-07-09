"""
配置读取工具。
这里专门处理 YAML 和 ROI JSON
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH_CONFIG_KEYS = {"weights", "roi_file", "save_video", "events_jsonl"}


def model_arg(path: str | Path) -> str:
    """把权重路径转成 Ultralytics 能识别的字符串。"""
    path = Path(path)
    return str(path.resolve()) if path.exists() else str(path)


def parse_roi(text: str | None) -> list[tuple[float, float]] | None:
    """解析命令行里直接写的 ROI，例如 'x1,y1;x2,y2;x3,y3'。"""
    if not text:
        return None
    points = []
    for pair in text.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    if len(points) < 3:
        raise ValueError("ROI needs at least 3 points, e.g. 100,120;520,120;560,420;80,420")
    return points


def parse_line(text: str | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """解析命令行里直接写的警戒线，例如 'x1,y1;x2,y2'。"""
    if not text:
        return None
    points = []
    for pair in text.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    if len(points) != 2:
        raise ValueError('Line needs exactly 2 points, e.g. "200,300;600,300"')
    return points[0], points[1]


def load_region_file(
    path: Path | None,
) -> tuple[str, list[tuple[float, float]] | None, tuple[tuple[float, float], tuple[float, float]] | None, str]:
    """从 roi_example.json 这类文件里读取禁区多边形和警戒线。"""
    if path is None:
        return "restricted", None, None, "warning_line"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "restricted", normalize_points(data), None, "warning_line"
    if not isinstance(data, dict):
        raise ValueError("ROI file must be a JSON object or a point list.")

    name = str(data.get("name", "restricted"))
    line_name = str(data.get("line_name", "warning_line"))
    points = data.get("points") or data.get("roi") or data.get("polygon")
    line_points = data.get("line")
    if points is None and "regions" in data:
        regions = data["regions"]
        if not regions:
            raise ValueError("ROI file regions is empty.")
        region = regions[0]
        name = str(region.get("name", name))
        points = region.get("points") or region.get("roi") or region.get("polygon")
    roi = normalize_points(points) if points is not None else None
    line = normalize_line_points(line_points) if line_points is not None else None
    return name, roi, line, line_name


def load_runtime_config(path: Path | None, valid_keys: set[str]) -> dict:
    """读取 edge_runtime.yaml，并作为 argparse 的默认值。

    注意：命令行参数优先级更高，所以临时测试时可以不用改 YAML。
    """
    if path is None or not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used. Install with: pip install PyYAML") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config must be a YAML object: {path}")

    defaults = {}
    for raw_key, value in data.items():
        key = str(raw_key).replace("-", "_")
        if key.startswith("_") or key.endswith("_comment") or key not in valid_keys or key == "config":
            continue
        if key in PATH_CONFIG_KEYS and value is not None:
            configured_path = Path(str(value))
            defaults[key] = configured_path if configured_path.is_absolute() else ROOT / configured_path
        else:
            defaults[key] = value
    return defaults


def normalize_points(points) -> list[tuple[float, float]]:
    """检查 ROI 点数量并统一成 float 坐标。"""
    parsed = [(float(point[0]), float(point[1])) for point in points]
    if len(parsed) < 3:
        raise ValueError("ROI needs at least 3 points.")
    return parsed


def normalize_line_points(points) -> tuple[tuple[float, float], tuple[float, float]]:
    """检查警戒线端点，必须刚好两个点。"""
    parsed = [(float(point[0]), float(point[1])) for point in points]
    if len(parsed) != 2:
        raise ValueError("Warning line needs exactly 2 points.")
    return parsed[0], parsed[1]
