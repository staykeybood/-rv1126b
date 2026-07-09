"""
异常行为规则模块。
输入是已经带 track_id 的人体轨迹，不再直接看 YOLO 原始检测框。
这里根据“同一个人”的位置变化判断入侵、停留、徘徊、越线、奔跑、遮挡和非工作时间出现。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .geometry import Point, bbox_bottom_center, distance, point_in_polygon


Line = tuple[Point, Point]


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


@dataclass
class BehaviorConfig:
    roi: list[Point] | None = None
    region_name: str = "restricted"
    intrusion_min_frames: int = 3

    loiter_seconds: float = 20.0
    stationary_speed_px_s: float = 12.0
    enable_full_frame_loitering: bool = False

    wandering_seconds: float = 30.0
    wandering_min_path_px: float = 150.0
    wandering_max_displacement_px: float = 200.0

    repeat_approach_window_seconds: float = 60.0
    repeat_approach_count: int = 3

    line: Line | None = None
    line_name: str = "warning_line"
    line_direction: str = "any"

    running_speed_px_s: float = 200.0
    running_min_frames: int = 8

    camera_block_ratio: float = 0.55
    camera_block_seconds: float = 3.0

    enable_after_hours: bool = True
    work_start: str = "08:00"
    work_end: str = "18:00"
    workdays: tuple[int, ...] = (1, 2, 3, 4, 5)

    max_history: int = 300
    stale_seconds: float = 3.0
    event_cooldown_seconds: float = 30.0
    after_hours_cooldown_seconds: float = 60.0


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    positions: deque[tuple[float, Point]] = field(default_factory=deque)
    in_roi_frames: int = 0
    roi_enter_time: float | None = None
    last_in_roi: bool = False
    approach_times: deque[float] = field(default_factory=deque)
    last_line_side: float | None = None
    running_frames: int = 0
    large_bbox_start: float | None = None
    last_event_time: dict[str, float] = field(default_factory=dict)


class BehaviorAnalyzer:
    """基于轨迹的安防规则分析器。"""

    def __init__(self, config: BehaviorConfig):
        self.config = config
        self.states: dict[int, TrackState] = {}
        self.global_last_event_time: dict[str, float] = {}

    def update(
        self,
        tracks: Sequence[Sequence[float]],
        timestamp: float,
        frame_shape: tuple[int, int] | None = None,
        wall_time: datetime | None = None,
    ) -> list[dict]:
        events: list[dict] = []
        after_hours = self.is_after_hours(wall_time or datetime.now())

        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            tid = int(track_id)
            bbox = (float(x1), float(y1), float(x2), float(y2))
            bottom = bbox_bottom_center(bbox)
            in_roi = point_in_polygon(bottom, self.config.roi)

            state = self.states.get(tid)
            if state is None:
                state = TrackState(first_seen=timestamp, last_seen=timestamp)
                self.states[tid] = state

            speed = self._instant_speed(state, timestamp, bottom)
            state.last_seen = timestamp
            state.positions.append((timestamp, bottom))
            while len(state.positions) > self.config.max_history:
                state.positions.popleft()

            roi_active = self.config.roi is not None
            entered_roi = roi_active and in_roi and not state.last_in_roi
            self._update_roi_state(state, in_roi, entered_roi, timestamp)

            if after_hours:
                event = self._cooldown_event("after_hours_intrusion", tid, timestamp)
                if event:
                    event.update(
                        {
                            "level": "critical",
                            "message": "非工作时间检测到人员",
                            "bbox": self._round_bbox(bbox),
                        }
                    )
                    events.append(event)

            event = self._intrusion_event(state, tid, timestamp, bbox, bottom)
            if event:
                events.append(event)

            event = self._loitering_event(state, tid, timestamp, bbox)
            if event:
                events.append(event)

            event = self._wandering_event(state, tid, timestamp, bbox)
            if event:
                events.append(event)

            event = self._repeat_approach_event(state, tid, timestamp, bbox, entered_roi)
            if event:
                events.append(event)

            event = self._line_crossing_event(state, tid, timestamp, bbox, bottom)
            if event:
                events.append(event)

            event = self._running_event(state, tid, timestamp, bbox, speed, in_roi, after_hours)
            if event:
                events.append(event)

            event = self._camera_block_event(state, tid, timestamp, bbox, frame_shape)
            if event:
                events.append(event)

            state.last_in_roi = in_roi

        self._remove_stale(timestamp)
        return events

    def counts(self, tracks: Sequence[Sequence[float]]) -> dict:
        person_ids = {int(track[4]) for track in tracks}
        if self.config.roi:
            roi_ids = {
                int(track[4])
                for track in tracks
                if point_in_polygon(bbox_bottom_center(track[:4]), self.config.roi)
            }
        else:
            roi_ids = person_ids
        return {"persons": len(person_ids), "roi_persons": len(roi_ids)}

    def is_after_hours(self, now: datetime) -> bool:
        if not self.config.enable_after_hours:
            return False
        if now.isoweekday() not in self.config.workdays:
            return True

        start_h, start_m = parse_hhmm(self.config.work_start)
        end_h, end_m = parse_hhmm(self.config.work_end)
        start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if start <= end:
            return not (start <= now < end)
        return end <= now < start

    def _update_roi_state(self, state: TrackState, in_roi: bool, entered_roi: bool, timestamp: float) -> None:
        if in_roi:
            state.in_roi_frames += 1
            if state.roi_enter_time is None:
                state.roi_enter_time = timestamp
        else:
            state.in_roi_frames = 0
            state.roi_enter_time = None

        if entered_roi:
            state.approach_times.append(timestamp)
        window_start = timestamp - self.config.repeat_approach_window_seconds
        while state.approach_times and state.approach_times[0] < window_start:
            state.approach_times.popleft()

    def _intrusion_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        bottom: Point,
    ) -> dict | None:
        if not self.config.roi or state.in_roi_frames < self.config.intrusion_min_frames:
            return None
        event = self._cooldown_event("intrusion", track_id, timestamp)
        if not event:
            return None
        event.update(
            {
                "level": "critical",
                "message": "人员进入禁区",
                "region": self.config.region_name,
                "bbox": self._round_bbox(bbox),
                "bottom_center": [round(bottom[0], 2), round(bottom[1], 2)],
                "frames": state.in_roi_frames,
            }
        )
        return event

    def _loitering_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
    ) -> dict | None:
        loiter_zone_active = self.config.roi is not None or self.config.enable_full_frame_loitering
        if not loiter_zone_active or state.roi_enter_time is None:
            return None
        duration = timestamp - state.roi_enter_time
        if duration < self.config.loiter_seconds:
            return None

        avg_speed = self._avg_speed(state, since=state.roi_enter_time)
        if avg_speed > self.config.stationary_speed_px_s:
            return None
        event = self._cooldown_event("loitering", track_id, timestamp)
        if not event:
            return None
        event.update(
            {
                "level": "warning",
                "message": "人员异常停留",
                "region": self.config.region_name if self.config.roi else "full_frame",
                "duration": round(duration, 2),
                "avg_speed": round(avg_speed, 2),
                "bbox": self._round_bbox(bbox),
            }
        )
        return event

    def _wandering_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
    ) -> dict | None:
        if not self.config.roi or state.roi_enter_time is None:
            return None
        duration = timestamp - state.roi_enter_time
        if duration < self.config.wandering_seconds:
            return None

        positions = self._positions_since(state, state.roi_enter_time)
        if len(positions) < 3:
            return None
        path = self._path_length(positions)
        displacement = distance(positions[0][1], positions[-1][1])
        if path < self.config.wandering_min_path_px:
            return None
        if displacement > self.config.wandering_max_displacement_px:
            return None
        event = self._cooldown_event("wandering", track_id, timestamp)
        if not event:
            return None
        event.update(
            {
                "level": "warning",
                "message": "人员在敏感区域徘徊",
                "region": self.config.region_name,
                "duration": round(duration, 2),
                "path_length": round(path, 2),
                "displacement": round(displacement, 2),
                "bbox": self._round_bbox(bbox),
            }
        )
        return event

    def _repeat_approach_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        entered_roi: bool,
    ) -> dict | None:
        if not entered_roi or len(state.approach_times) < self.config.repeat_approach_count:
            return None
        event = self._cooldown_event("repeat_approach", track_id, timestamp)
        if not event:
            return None
        event.update(
            {
                "level": "warning",
                "message": "人员多次靠近敏感区域",
                "region": self.config.region_name,
                "count": len(state.approach_times),
                "window_seconds": self.config.repeat_approach_window_seconds,
                "bbox": self._round_bbox(bbox),
            }
        )
        return event

    def _line_crossing_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        point: Point,
    ) -> dict | None:
        if self.config.line is None:
            return None
        side = signed_line_side(point, self.config.line)
        if abs(side) < 1e-6:
            return None

        if state.last_line_side is None:
            state.last_line_side = side
            return None

        crossed = side * state.last_line_side < 0
        old_side = state.last_line_side
        state.last_line_side = side
        if not crossed or not self._line_direction_allowed(old_side, side):
            return None

        event = self._cooldown_event("line_crossing", track_id, timestamp)
        if not event:
            return None
        event.update(
            {
                "level": "critical",
                "message": "人员穿越警戒线",
                "line": self.config.line_name,
                "direction": "positive_to_negative" if old_side > 0 > side else "negative_to_positive",
                "bbox": self._round_bbox(bbox),
                "point": [round(point[0], 2), round(point[1], 2)],
            }
        )
        return event

    def _running_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        speed: float,
        in_roi: bool,
        after_hours: bool,
    ) -> dict | None:
        if speed >= self.config.running_speed_px_s:
            state.running_frames += 1
        else:
            state.running_frames = 0
        if state.running_frames < self.config.running_min_frames:
            return None

        event = self._cooldown_event("running", track_id, timestamp)
        if not event:
            return None
        critical = (self.config.roi is not None and in_roi) or after_hours
        event.update(
            {
                "level": "critical" if critical else "warning",
                "message": "检测到人员快速奔跑",
                "speed": round(speed, 2),
                "frames": state.running_frames,
                "region": self.config.region_name if self.config.roi and in_roi else "full_frame",
                "bbox": self._round_bbox(bbox),
            }
        )
        return event

    def _camera_block_event(
        self,
        state: TrackState,
        track_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        frame_shape: tuple[int, int] | None,
    ) -> dict | None:
        if frame_shape is None:
            return None
        height, width = frame_shape[:2]
        frame_area = max(float(width * height), 1.0)
        x1, y1, x2, y2 = bbox
        cover_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
        if cover_ratio >= self.config.camera_block_ratio:
            if state.large_bbox_start is None:
                state.large_bbox_start = timestamp
        else:
            state.large_bbox_start = None
            return None

        duration = timestamp - state.large_bbox_start
        if duration < self.config.camera_block_seconds:
            return None
        event = self._cooldown_event("camera_block", track_id, timestamp)
        if not event:
            return None
        event.update(
            {
                "level": "critical",
                "message": "疑似遮挡或贴近摄像头",
                "duration": round(duration, 2),
                "cover_ratio": round(cover_ratio, 3),
                "bbox": self._round_bbox(bbox),
            }
        )
        return event

    def _line_direction_allowed(self, old_side: float, new_side: float) -> bool:
        direction = self.config.line_direction
        if direction == "any":
            return True
        if direction == "positive_to_negative":
            return old_side > 0 > new_side
        if direction == "negative_to_positive":
            return old_side < 0 < new_side
        return True

    def _instant_speed(self, state: TrackState, timestamp: float, point: Point) -> float:
        if not state.positions:
            return 0.0
        prev_t, prev_p = state.positions[-1]
        dt = max(timestamp - prev_t, 1e-6)
        return distance(prev_p, point) / dt

    def _cooldown_event(self, event_type: str, track_id: int, timestamp: float) -> dict | None:
        state = self.states[track_id]
        if event_type == "after_hours_intrusion":
            last_global = self.global_last_event_time.get(event_type, -1e9)
            if timestamp - last_global < self.config.after_hours_cooldown_seconds:
                return None
            self.global_last_event_time[event_type] = timestamp

        last = state.last_event_time.get(event_type, -1e9)
        cooldown = (
            self.config.after_hours_cooldown_seconds
            if event_type == "after_hours_intrusion"
            else self.config.event_cooldown_seconds
        )
        if timestamp - last < cooldown:
            return None
        state.last_event_time[event_type] = timestamp
        return {
            "type": event_type,
            "type_cn": EVENT_LABELS_CN.get(event_type, event_type),
            "track_id": track_id,
            "time": round(timestamp, 3),
        }

    def _avg_speed(self, state: TrackState, since: float | None = None) -> float:
        positions = self._positions_since(state, since)
        if len(positions) < 2:
            return 0.0
        dt = max(positions[-1][0] - positions[0][0], 1e-6)
        return self._path_length(positions) / dt

    def _positions_since(self, state: TrackState, since: float | None = None) -> list[tuple[float, Point]]:
        positions = list(state.positions)
        if since is not None:
            positions = [(t, p) for t, p in positions if t >= since]
        return positions

    def _path_length(self, positions: list[tuple[float, Point]]) -> float:
        path = 0.0
        prev = positions[0][1]
        for _, point in positions[1:]:
            path += distance(prev, point)
            prev = point
        return path

    def _remove_stale(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, state in self.states.items()
            if timestamp - state.last_seen > self.config.stale_seconds
        ]
        for track_id in stale:
            del self.states[track_id]

    def _round_bbox(self, bbox: tuple[float, float, float, float]) -> list[float]:
        return [round(float(v), 2) for v in bbox]


def signed_line_side(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def parse_hhmm(text: str) -> tuple[int, int]:
    hour, minute = text.split(":", 1)
    return int(hour), int(minute)
