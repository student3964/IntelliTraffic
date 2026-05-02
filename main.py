"""
============================================================
IntelliTraffic – Main Pipeline Runner
============================================================
Orchestrates the full traffic monitoring pipeline:
    1. Load configuration
    2. Initialize all modules
    3. Open video source
    4. Frame-by-frame processing loop:
        - Detect vehicles & persons (YOLOv8)
        - Track objects (DeepSORT)
        - Estimate speeds (perspective transform)
        - Analyze density (ROI counting + heatmap)
        - Detect violations (overspeeding, triple riding)
        - Recognize plates (ANPR / EasyOCR)
        - Generate e-challans (enforcement)
        - Update risk score
        - Draw visualizations
    5. Cleanup and save final reports

Usage:
    python main.py --source sample.mp4
    python main.py --source 0  (webcam)
    python main.py --config custom_config.yaml --source video.mp4 --output result.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.drawing import (
    draw_tracked_vehicles, draw_roi, draw_density_heatmap,
    draw_violations, draw_dashboard_overlay, draw_speed_lines
)
from detection.detector import VehicleDetector
from detection.tracker import VehicleTracker
from modules.speed import SpeedEstimator
from modules.density import DensityAnalyzer
from modules.violation import ViolationDetector
from modules.anpr import PlateRecognizer
from engine.risk_engine import RiskEngine
from engine.enforcement import EnforcementEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="IntelliTraffic – Traffic Intelligence & Automated Enforcement",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Video file path or camera index (0 for webcam)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output video")
    parser.add_argument("--display", action="store_true", default=None,
                        help="Show live display window")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display window")
    return parser.parse_args()


def open_video_source(source):
    """
    Open video source (file or camera).

    Args:
        source: File path string or camera index (int or '0').

    Returns:
        cv2.VideoCapture object.
    """
    # Check if source is a camera index
    try:
        source_int = int(source)
        print(f"[Pipeline] Opening camera index: {source_int}")
        cap = cv2.VideoCapture(source_int)
    except (ValueError, TypeError):
        print(f"[Pipeline] Opening video file: {source}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    # Print video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Pipeline] Video: {width}x{height} @ {fps:.1f} FPS, {total} frames")

    return cap


def run_pipeline(config_path: str = "config.yaml", source: str = None,
                 output: str = None, display: bool = True):
    """
    Run the complete IntelliTraffic pipeline.

    Args:
        config_path: Path to configuration YAML.
        source: Video source override.
        output: Output video path override.
        display: Whether to show live display.
    """
    # ── 1. Load Configuration ──────────────────────────────
    print("=" * 60)
    print("  IntelliTraffic – Traffic Intelligence System")
    print("=" * 60)

    config = Config(config_path)

    # Override config with CLI args
    video_source = source or config.get("video", "source", default="sample.mp4")
    output_path = output or config.get("video", "output_path", default="output.mp4")
    resize_width = config.get("video", "resize_width", default=None)

    # ── 2. Initialize All Modules ──────────────────────────
    print("\n[Pipeline] Initializing modules...")

    detector = VehicleDetector(config)
    tracker = VehicleTracker(config)
    speed_estimator = SpeedEstimator(config)
    density_analyzer = DensityAnalyzer(config)
    violation_detector = ViolationDetector(config)
    plate_recognizer = PlateRecognizer(config)
    risk_engine = RiskEngine(config)
    enforcement = EnforcementEngine(config)

    print("[Pipeline] All modules initialized.\n")

    # ── 3. Open Video Source ───────────────────────────────
    cap = open_video_source(video_source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time_delta = 1.0 / fps

    # Setup output video writer
    writer = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if resize_width and width > 0:
            height = int(height * resize_width / width)
            width = resize_width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Output video: {output_path}")

    # ── 4. Processing Loop ─────────────────────────────────
    frame_count = 0
    start_time = time.time()
    current_time = 0.0  # Simulated time based on FPS

    print("[Pipeline] Starting processing loop... (Press 'q' to quit)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[Pipeline] End of video / no more frames.")
                break

            frame_count += 1
            current_time = frame_count * frame_time_delta

            # Resize if configured
            if resize_width and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # ── 4a. Detect vehicles & persons ──────────────
            detections = detector.detect(frame)
            vehicles = detections["vehicles"]
            persons = detections["persons"]
            all_dets = vehicles + persons

            # ── 4b. Track objects ──────────────────────────
            tracked = tracker.update(all_dets, frame, frame_time=current_time)

            # ── 4c. Estimate speeds ────────────────────────
            speeds = speed_estimator.estimate_speeds(
                tracked, tracker.track_history
            )

            # ── 4d. Analyze density ────────────────────────
            density = density_analyzer.analyze(tracked, frame.shape[:2])

            # ── 4e. Detect violations ──────────────────────
            violations = violation_detector.detect(
                tracked, persons, speeds, tracker.track_history
            )

            # ── 4f. ANPR on violating vehicles ─────────────
            plates = {}
            if violations:
                # Run ANPR specifically for violating vehicles
                violation_objects = [
                    obj for obj in tracked
                    if obj["track_id"] in {v["track_id"] for v in violations}
                ]
                plates = plate_recognizer.recognize(frame, violation_objects)

            # ── 4g. Generate e-challans ────────────────────
            for v in violations:
                plate = plates.get(v["track_id"])
                enforcement.process_violation(v, frame, plate)

            # ── 4h. Update risk score ──────────────────────
            risk = risk_engine.update(
                density["count"],
                speed_estimator.get_average_speed(),
                violation_detector.get_violation_count()
            )

            # ── 4i. Draw visualizations ────────────────────
            display_frame = frame.copy()

            # Draw ROI region
            draw_roi(display_frame, density_analyzer.get_roi_points().tolist())

            # Draw speed reference lines
            draw_speed_lines(display_frame, speed_estimator.get_source_points())

            # Draw tracked vehicles with speeds
            draw_tracked_vehicles(display_frame, tracked, speeds)

            # Draw violations (red highlights)
            draw_violations(display_frame, violations)

            # Draw heatmap overlay (if enabled)
            heatmap = density_analyzer.get_heatmap()
            if heatmap is not None and config.get("density", "heatmap_enabled", default=True):
                display_frame = draw_density_heatmap(display_frame, heatmap, alpha=0.3)

            # Draw stats overlay
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            stats = {
                "vehicle_count": density["count"],
                "avg_speed": speed_estimator.get_average_speed(),
                "density_level": density["level"],
                "risk_level": risk.get("level", "LOW"),
                "violation_count": violation_detector.get_violation_count(),
                "fps": current_fps
            }
            draw_dashboard_overlay(display_frame, stats)

            # ── 4j. Output ─────────────────────────────────
            if writer:
                writer.write(display_frame)

            if display:
                cv2.imshow("IntelliTraffic", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[Pipeline] User quit (pressed 'q').")
                    break

            # Progress logging (every 100 frames)
            if frame_count % 100 == 0:
                print(f"  Frame {frame_count} | FPS: {current_fps:.1f} | "
                      f"Vehicles: {density['count']} | "
                      f"Density: {density['level']} | "
                      f"Risk: {risk.get('level', 'LOW')} | "
                      f"Violations: {violation_detector.get_violation_count()}")

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")

    finally:
        # ── 5. Cleanup ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("  Pipeline Summary")
        print("=" * 60)
        total_time = time.time() - start_time
        print(f"  Frames processed: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {frame_count / total_time:.1f}" if total_time > 0 else "")
        print(f"  Violations detected: {violation_detector.get_violation_count()}")
        print(f"  Challans issued: {enforcement.get_challan_count()}")

        v_stats = violation_detector.get_stats()
        if v_stats["by_type"]:
            print(f"  Violation breakdown: {v_stats['by_type']}")

        print(f"  Challan logs: {enforcement.csv_path}")
        print("=" * 60)

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    show_display = True
    if args.no_display:
        show_display = False
    elif args.display is not None:
        show_display = args.display

    run_pipeline(
        config_path=args.config,
        source=args.source,
        output=args.output,
        display=show_display
    )
