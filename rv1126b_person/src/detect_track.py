"""
主运行入口。
调用 YOLO 做人体检测、交给跟踪器分配 ID，再把轨迹送进异常行为规则。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from edge_person.behavior import BehaviorAnalyzer, BehaviorConfig
from edge_person.config import load_region_file, load_runtime_config, model_arg, parse_line, parse_roi
from edge_person.enhance import enhance_frame
from edge_person.reporter import EventReporter
from edge_person.tracker import LightByteTracker
from edge_person.video import (
    WINDOW_NAME,
    detections_from_result,
    get_current_window_size,
    make_display_canvas,
    open_source,
    resolve_display_size,
    transform_line_for_display,
    transform_points_for_display,
    transform_tracks_for_display,
)
from edge_person.visual import draw, update_recent_events


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    """准备运行参数。

    平时主要改 configs/edge_runtime.yaml；命令行参数适合临时覆盖某一项。
    比如 YAML 里 source=0，运行时加 --source demo.mp4 就会临时改成视频文件。
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "edge_runtime.yaml",
        help="YAML runtime config; CLI arguments override values from this file.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="YOLOv8 person detection, tracking, and edge behavior events.",
        parents=[pre_parser],
    )

    # 模型和输入源：权重、摄像头编号、视频流地址
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default="0", help="Camera index, video path, or RTSP/HTTP stream.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default=None, help="Example: 0, cuda:0, or cpu.")
    parser.add_argument("--process-every", type=int, default=1, help="Run detection every N frames to improve edge FPS.")
    parser.add_argument("--camera-width", type=int, default=0, help="Set camera capture width when source is a camera.")
    parser.add_argument("--camera-height", type=int, default=0, help="Set camera capture height when source is a camera.")
    parser.add_argument("--camera-fps", type=float, default=0.0, help="Set camera FPS when source is a camera.")

    # 显示、日志和检测阈值：影响画面窗口、FPS 打印、YOLO 置信度等。
    parser.add_argument("--fps-log-interval", type=float, default=2.0, help="Print runtime FPS every N seconds; 0 disables.")
    parser.add_argument("--hide-fps", action="store_true")
    parser.add_argument("--display-width", type=int, default=0, help="Preview window width; 0 uses camera frame width.")
    parser.add_argument("--display-height", type=int, default=0, help="Preview window height; 0 uses camera frame height.")
    parser.add_argument("--max-alert-rows", type=int, default=3, help="Maximum alert rows shown in the preview window.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)

    # 跟踪参数：这些主要影响 ID 是否稳定，调得太激进会让同一个人频繁换 ID。
    parser.add_argument("--track-thresh", type=float, default=0.25)
    parser.add_argument("--low-thresh", type=float, default=0.10)
    parser.add_argument("--new-track-thresh", type=float, default=0.30)
    parser.add_argument("--track-iou", type=float, default=0.15)
    parser.add_argument("--max-age", type=int, default=90)
    parser.add_argument("--min-hits", type=int, default=1)

    # ROI、警戒线和异常规则阈值：区域入侵、停留、徘徊、越线、奔跑都看这些。
    parser.add_argument("--roi", default=None, help='Polygon: "x1,y1;x2,y2;x3,y3".')
    parser.add_argument("--roi-file", type=Path, default=None, help="JSON file with ROI name and points.")
    parser.add_argument("--line", default=None, help='Warning line: "x1,y1;x2,y2".')
    parser.add_argument("--line-direction", default="any", choices=["any", "positive_to_negative", "negative_to_positive"])
    parser.add_argument("--intrusion-min-frames", type=int, default=3)
    parser.add_argument("--loiter-seconds", type=float, default=20.0)
    parser.add_argument("--stationary-speed", type=float, default=12.0, help="Pixel/s threshold for loitering.")
    parser.add_argument("--wandering-seconds", type=float, default=30.0)
    parser.add_argument("--wandering-min-path", type=float, default=150.0)
    parser.add_argument("--wandering-max-displacement", type=float, default=200.0)
    parser.add_argument("--repeat-approach-window", type=float, default=60.0)
    parser.add_argument("--repeat-approach-count", type=int, default=3)
    parser.add_argument("--running-speed", type=float, default=200.0)
    parser.add_argument("--running-min-frames", type=int, default=8)
    parser.add_argument("--camera-block-ratio", type=float, default=0.55)
    parser.add_argument("--camera-block-seconds", type=float, default=3.0)
    parser.add_argument("--work-start", default="08:00")
    parser.add_argument("--work-end", default="18:00")
    parser.add_argument("--workdays", default="1,2,3,4,5", help="ISO weekdays, 1=Mon ... 7=Sun.")
    parser.add_argument("--disable-after-hours", action="store_true")
    parser.add_argument("--loiter-full-frame", action="store_true", help="Enable loitering rules when no ROI is set.")
    parser.add_argument("--event-cooldown", type=float, default=30.0)
    parser.add_argument("--after-hours-cooldown", type=float, default=60.0)

    # 图像增强、保存视频、事件上报：板端不需要显示时也可以只保留上报。
    parser.add_argument("--enhance", action="store_true", help="Enable gamma/CLAHE enhancement before inference.")
    parser.add_argument("--denoise", action="store_true", help="Use with --enhance for noisy weak-light images.")
    parser.add_argument("--save-video", type=Path, default=None)
    parser.add_argument("--events-jsonl", type=Path, default=ROOT / "runs" / "track" / "events.jsonl")
    parser.add_argument("--udp", default=None, help="Send JSON events to host:port by UDP.")
    parser.add_argument("--tcp", default=None, help="Send JSON events to host:port by TCP.")
    parser.add_argument("--serial-port", default=None, help="Example: COM3 or /dev/ttyS3.")
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument("--summary-interval", type=float, default=1.0)
    parser.add_argument("--english-ui", action="store_true", help="Use English overlay text when Chinese fonts are unavailable or too slow.")
    parser.add_argument("--show", action="store_true")

    parser.set_defaults(**load_runtime_config(pre_args.config, {action.dest for action in parser._actions}))
    return parser


def build_analyzer(args, roi, line, line_name: str, region_name: str) -> BehaviorAnalyzer:
    """把配置参数整理成行为分析器需要的格式。"""
    workdays = tuple(int(item) for item in args.workdays.split(",") if item.strip())
    return BehaviorAnalyzer(
        BehaviorConfig(
            roi=roi,
            region_name=region_name,
            intrusion_min_frames=args.intrusion_min_frames,
            loiter_seconds=args.loiter_seconds,
            stationary_speed_px_s=args.stationary_speed,
            wandering_seconds=args.wandering_seconds,
            wandering_min_path_px=args.wandering_min_path,
            wandering_max_displacement_px=args.wandering_max_displacement,
            repeat_approach_window_seconds=args.repeat_approach_window,
            repeat_approach_count=args.repeat_approach_count,
            line=line,
            line_name=line_name,
            line_direction=args.line_direction,
            running_speed_px_s=args.running_speed,
            running_min_frames=args.running_min_frames,
            camera_block_ratio=args.camera_block_ratio,
            camera_block_seconds=args.camera_block_seconds,
            enable_after_hours=not args.disable_after_hours,
            work_start=args.work_start,
            work_end=args.work_end,
            workdays=workdays,
            enable_full_frame_loitering=args.loiter_full_frame,
            event_cooldown_seconds=args.event_cooldown,
            after_hours_cooldown_seconds=args.after_hours_cooldown,
        )
    )


def prepare_capture(args):
    """打开摄像头/视频，并按配置准备保存视频或显示窗口。"""
    cap = open_source(args.source)
    if args.source.isdigit():
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_fps = max(fps / args.process_every, 1.0)
        writer = cv2.VideoWriter(str(args.save_video), fourcc, writer_fps, (width, height))

    display_size = None
    if args.show:
        display_size = resolve_display_size(width, height, args.display_width, args.display_height)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *display_size)

    return cap, writer, fps, display_size


def draw_for_window(frame, tracks, counts, recent_events, roi, line, runtime_fps, args, display_size):
    """给窗口显示用的绘制函数。
    """
    canvas_width, canvas_height = get_current_window_size(*display_size)
    display_frame, display_scale, display_offset = make_display_canvas(frame, canvas_width, canvas_height)
    display_tracks = transform_tracks_for_display(tracks, display_scale, display_offset)
    display_roi = transform_points_for_display(roi, display_scale, display_offset)
    display_line = transform_line_for_display(line, display_scale, display_offset)
    draw(
        display_frame,
        display_tracks,
        counts,
        recent_events,
        display_roi,
        display_line,
        None if args.hide_fps else runtime_fps,
        args.process_every,
        chinese_ui=not args.english_ui,
        max_alert_rows=args.max_alert_rows,
    )
    return display_frame


def main() -> None:
    # 1. 读取配置，创建模型、跟踪器、规则分析器和上报器。
    args = build_parser().parse_args()
    args.source = str(args.source)
    args.process_every = max(1, args.process_every)

    from ultralytics import YOLO

    model = YOLO(model_arg(args.weights))
    tracker = LightByteTracker(
        track_thresh=args.track_thresh,
        low_thresh=args.low_thresh,
        new_track_thresh=args.new_track_thresh,
        iou_threshold=args.track_iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )
    region_name, file_roi, file_line, line_name = load_region_file(args.roi_file)
    roi = parse_roi(args.roi) or file_roi
    line = parse_line(args.line) or file_line
    analyzer = build_analyzer(args, roi, line, line_name, region_name)
    reporter = EventReporter(args.events_jsonl, args.udp, args.tcp, args.serial_port, args.serial_baud)
    cap, writer, fps, display_size = prepare_capture(args)

    frame_idx = 0
    start = time.time()
    next_summary = 0.0
    recent_events: list[dict] = []
    runtime_fps: float | None = None
    processed_frames = 0
    fps_window_start = time.time()
    last_fps_log = fps_window_start

    # 2. 主循环：取帧 -> 检测 -> 跟踪 -> 规则判断 -> 上报/显示。
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % args.process_every != 0:
                continue
            timestamp = frame_idx / fps if not args.source.isdigit() else time.time() - start

            # 弱光/逆光场景可以开启 enhance；默认关闭，避免板端额外耗时。
            infer_frame = enhance_frame(frame, enable_denoise=args.denoise) if args.enhance else frame
            result = model.predict(
                infer_frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.nms_iou,
                classes=[0],
                device=args.device,
                verbose=False,
            )[0]
            # 检测框只说明“这一帧有人”，跟踪后的 track_id 才能判断停留、越线、奔跑。
            detections = detections_from_result(result)
            tracks = tracker.update(detections, frame_gap=args.process_every)
            events = analyzer.update(tracks, timestamp, frame_shape=frame.shape[:2])
            counts = analyzer.counts(tracks)

            for event in events:
                reporter.send(event)
                print(event)
            if events:
                update_recent_events(recent_events, events)

            if timestamp >= next_summary:
                reporter.send({"type": "count", "type_cn": "人数统计", "time": round(timestamp, 3), **counts})
                next_summary = timestamp + args.summary_interval

            processed_frames += 1
            now = time.time()
            elapsed = max(now - fps_window_start, 1e-6)
            instant_fps = processed_frames / elapsed
            runtime_fps = instant_fps if runtime_fps is None else 0.85 * runtime_fps + 0.15 * instant_fps
            if args.fps_log_interval > 0 and now - last_fps_log >= args.fps_log_interval:
                print(f"runtime_fps={runtime_fps:.2f}, process_every={args.process_every}, imgsz={args.imgsz}")
                last_fps_log = now
                fps_window_start = now
                processed_frames = 0

            # 保存视频时写入的是带框和告警栏的画面，方便离线检查效果。
            if writer is not None:
                output_frame = frame.copy()
                draw(
                    output_frame,
                    tracks,
                    counts,
                    recent_events,
                    roi,
                    line,
                    None if args.hide_fps else runtime_fps,
                    args.process_every,
                    chinese_ui=not args.english_ui,
                    max_alert_rows=args.max_alert_rows,
                )
                writer.write(output_frame)

            # 板端如果没有接屏幕，可以不加 --show，只通过 JSON/串口/网络拿报警。
            if args.show:
                display_frame = draw_for_window(frame, tracks, counts, recent_events, roi, line, runtime_fps, args, display_size)
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        reporter.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
