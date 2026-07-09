"""
轻量跟踪模块。
这个跟踪器不使用 ReID 网络，只靠框重叠和速度预测维持 ID，
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = a[:, None, :4]
    b = b[None, :, :4]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def center(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox[:4]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


@dataclass
class Track:
    bbox: np.ndarray
    score: float
    track_id: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    age: int = 0
    hits: int = 1
    time_since_update: int = 0

    def predict(self, steps: int = 1) -> np.ndarray:
        steps = max(1, int(steps))
        self.age += 1
        self.time_since_update += steps
        self.bbox[[0, 2]] += self.velocity[0] * steps
        self.bbox[[1, 3]] += self.velocity[1] * steps
        return self.bbox.copy()

    def update(self, det: np.ndarray, momentum: float = 0.65) -> None:
        old_center = center(self.bbox)
        new_center = center(det)
        measured_velocity = new_center - old_center
        self.velocity = momentum * self.velocity + (1.0 - momentum) * measured_velocity
        self.bbox = det[:4].astype(np.float32)
        self.score = float(det[4])
        self.hits += 1
        self.time_since_update = 0


class LightByteTracker:
    """轻量 ByteTrack 风格跟踪器，额外加了一点速度预测。

    高置信度框负责新建轨迹，低置信度框用来短时间续住旧轨迹，
    尽量在不卡板子的前提下保持 track_id 稳定。
    """

    def __init__(
        self,
        track_thresh: float = 0.35,
        low_thresh: float = 0.10,
        new_track_thresh: float = 0.45,
        iou_threshold: float = 0.25,
        second_iou_threshold: float = 0.15,
        max_age: int = 30,
        min_hits: int = 3,
        velocity_weight: float = 0.08,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.iou_threshold = iou_threshold
        self.second_iou_threshold = second_iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.velocity_weight = velocity_weight
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, detections: np.ndarray, frame_gap: int = 1) -> np.ndarray:
        self.frame_count += 1
        detections = self._prepare_detections(detections)

        for track in self.tracks:
            track.predict(frame_gap)

        high = detections[detections[:, 4] >= self.track_thresh]
        low = detections[(detections[:, 4] >= self.low_thresh) & (detections[:, 4] < self.track_thresh)]

        matches, unmatched_high, unmatched_tracks = self._associate(high, list(range(len(self.tracks))), self.iou_threshold)
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(high[det_idx])

        second_matches, _, unmatched_tracks = self._associate(
            low,
            unmatched_tracks,
            self.second_iou_threshold,
        )
        for det_idx, track_idx in second_matches:
            self.tracks[track_idx].update(low[det_idx])

        for det_idx in unmatched_high:
            det = high[det_idx]
            if det[4] >= self.new_track_thresh:
                self._start_track(det)

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        return self._visible_tracks()

    def _prepare_detections(self, detections: np.ndarray) -> np.ndarray:
        if detections is None or detections.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] < 5:
            raise ValueError("detections must have shape Nx5: x1,y1,x2,y2,score")
        detections = detections[:, :5]
        valid = (
            (detections[:, 2] > detections[:, 0])
            & (detections[:, 3] > detections[:, 1])
            & (detections[:, 4] >= self.low_thresh)
        )
        return detections[valid]

    def _start_track(self, det: np.ndarray) -> None:
        self.tracks.append(
            Track(
                bbox=det[:4].astype(np.float32),
                score=float(det[4]),
                track_id=self._next_id,
            )
        )
        self._next_id += 1

    def _visible_tracks(self) -> np.ndarray:
        outputs = []
        for track in self.tracks:
            visible_now = track.time_since_update == 0
            stable = track.hits >= self.min_hits or self.frame_count <= self.min_hits
            if visible_now and stable:
                outputs.append([*track.bbox.tolist(), float(track.track_id), float(track.score)])
        if not outputs:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(outputs, dtype=np.float32)

    def _associate(
        self,
        detections: np.ndarray,
        track_indices: list[int],
        iou_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(detections) == 0:
            return [], [], track_indices
        if not track_indices:
            return [], list(range(len(detections))), []

        track_boxes = np.asarray([self.tracks[idx].bbox for idx in track_indices], dtype=np.float32)
        ious = iou_matrix(detections[:, :4], track_boxes)
        scores = ious + self._velocity_scores(detections, track_indices)
        assignment = linear_assignment(-scores)

        matched: list[tuple[int, int]] = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(track_indices)
        for det_idx, local_track_idx in assignment:
            global_track_idx = track_indices[int(local_track_idx)]
            if ious[int(det_idx), int(local_track_idx)] < iou_threshold:
                continue
            matched.append((int(det_idx), global_track_idx))
            unmatched_dets.discard(int(det_idx))
            unmatched_tracks.discard(global_track_idx)

        return matched, sorted(unmatched_dets), sorted(unmatched_tracks)

    def _velocity_scores(self, detections: np.ndarray, track_indices: list[int]) -> np.ndarray:
        scores = np.zeros((len(detections), len(track_indices)), dtype=np.float32)
        for det_idx, det in enumerate(detections):
            det_center = center(det)
            for local_idx, track_idx in enumerate(track_indices):
                track = self.tracks[track_idx]
                speed = np.linalg.norm(track.velocity)
                if speed < 1e-3:
                    continue
                direction = det_center - center(track.bbox)
                norm = np.linalg.norm(direction)
                if norm < 1e-3:
                    continue
                cosine = float(np.dot(direction / norm, track.velocity / speed))
                scores[det_idx, local_idx] = self.velocity_weight * max(cosine, 0.0)
        return scores


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost_matrix)
        return np.stack([rows, cols], axis=1)
    except Exception:
        return greedy_assignment(cost_matrix)


def greedy_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    pairs = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = [
        (cost_matrix[row, col], row, col)
        for row in range(cost_matrix.shape[0])
        for col in range(cost_matrix.shape[1])
    ]
    for _, row, col in sorted(flat, key=lambda item: item[0]):
        if row in used_rows or col in used_cols:
            continue
        pairs.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return np.asarray(pairs, dtype=int)
