"""
============================================================
IntelliTraffic – Visualization & Drawing Utilities
============================================================
Helper functions for drawing bounding boxes, overlays,
ROI regions, heatmaps, and dashboard info panels on frames.
"""

import cv2
import numpy as np


# ── Color Palette (BGR format for OpenCV) ──────────────────
COLORS = {
    "car": (255, 178, 50),       # Orange
    "motorcycle": (0, 255, 127),  # Spring green
    "bus": (255, 100, 100),       # Light blue
    "truck": (147, 20, 255),      # Pink
    "person": (0, 255, 255),      # Yellow
    "default": (200, 200, 200),   # Gray
    "violation": (0, 0, 255),     # Red
    "roi": (0, 255, 0),           # Green
    "speed_ok": (0, 200, 0),      # Green
    "speed_warn": (0, 165, 255),  # Orange
    "speed_over": (0, 0, 255),    # Red
    "text_bg": (0, 0, 0),         # Black
    "risk_low": (0, 200, 0),      # Green
    "risk_medium": (0, 200, 255), # Yellow-orange
    "risk_high": (0, 0, 255),     # Red
}


def get_vehicle_color(class_name: str) -> tuple:
    """Get the color for a given vehicle class."""
    return COLORS.get(class_name, COLORS["default"])


def draw_label(frame, text: str, position: tuple, color: tuple = (255, 255, 255),
               bg_color: tuple = (0, 0, 0), font_scale: float = 0.5,
               thickness: int = 1, padding: int = 4):
    """
    Draw a text label with a filled background rectangle.

    Args:
        frame: Image to draw on (modified in place).
        text: Label text string.
        position: (x, y) top-left corner of the label.
        color: Text color (BGR).
        bg_color: Background rectangle color (BGR).
        font_scale: Font scale factor.
        thickness: Text thickness.
        padding: Padding around text inside the background rectangle.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Ensure label stays within frame bounds
    y = max(y, text_h + padding * 2)

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y - text_h - padding * 2),
        (x + text_w + padding * 2, y),
        bg_color, -1
    )
    # Draw text
    cv2.putText(
        frame, text,
        (x + padding, y - padding),
        font, font_scale, color, thickness, cv2.LINE_AA
    )


def draw_tracked_vehicles(frame, tracked_objects: list, speeds: dict = None):
    """
    Draw bounding boxes with track IDs, class names, and speeds.

    Args:
        frame: Image to draw on.
        tracked_objects: List of tracked vehicle dicts with keys:
            track_id, bbox (x1,y1,x2,y2), class_name, confidence
        speeds: Dict mapping track_id → speed in km/h.
    """
    for obj in tracked_objects:
        track_id = obj["track_id"]
        x1, y1, x2, y2 = [int(c) for c in obj["bbox"]]
        class_name = obj.get("class_name", "vehicle")
        confidence = obj.get("confidence", 0.0)
        color = get_vehicle_color(class_name)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label text
        label = f"ID:{track_id} {class_name}"
        if speeds and track_id in speeds:
            speed = speeds[track_id]
            label += f" {speed:.0f}km/h"
            # Color-code speed
            if speed > 80:
                color = COLORS["speed_over"]
            elif speed > 60:
                color = COLORS["speed_warn"]

        # Draw label above bounding box
        draw_label(frame, label, (x1, y1), color=(255, 255, 255),
                   bg_color=color, font_scale=0.5, thickness=1)

    return frame


def draw_roi(frame, roi_points: list, color: tuple = None, alpha: float = 0.2):
    """
    Draw a semi-transparent ROI polygon overlay.

    Args:
        frame: Image to draw on.
        roi_points: List of [x, y] polygon vertices.
        color: Fill color (BGR). Defaults to green.
        alpha: Transparency (0 = invisible, 1 = opaque).
    """
    if not roi_points:
        return frame

    if color is None:
        color = COLORS["roi"]

    pts = np.array(roi_points, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)

    return frame


def draw_density_heatmap(frame, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Overlay a colored heatmap on the frame.

    Args:
        frame: Base image.
        heatmap: 2D float array (same size as frame) with intensity values.
        alpha: Blend factor.

    Returns:
        Blended frame with heatmap overlay.
    """
    if heatmap is None or heatmap.max() == 0:
        return frame

    # Normalize heatmap to 0-255
    heatmap_norm = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8)

    # Apply colormap (COLORMAP_JET: blue → green → yellow → red)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Blend with original frame
    result = cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)
    return result


def draw_violations(frame, violations: list):
    """
    Highlight violations with red bounding boxes and labels.

    Args:
        frame: Image to draw on.
        violations: List of violation dicts with keys:
            track_id, bbox, violation_type, details
    """
    for v in violations:
        if "bbox" not in v:
            continue

        x1, y1, x2, y2 = [int(c) for c in v["bbox"]]
        color = COLORS["violation"]

        # Draw thick red border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw violation label
        label = f"⚠ {v['violation_type']}"
        if "details" in v:
            label += f" | {v['details']}"

        draw_label(frame, label, (x1, y1 - 25), color=(255, 255, 255),
                   bg_color=(0, 0, 200), font_scale=0.55, thickness=2)

    return frame


def draw_dashboard_overlay(frame, stats: dict):
    """
    Draw a stats panel overlay in the top-right corner of the frame.

    Args:
        frame: Image to draw on.
        stats: Dict with keys like:
            vehicle_count, avg_speed, density_level, risk_level,
            violation_count, fps
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 280
    panel_h = 200
    margin = 10
    x_start = w - panel_w - margin
    y_start = margin

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start + panel_w, y_start + panel_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Border
    cv2.rectangle(frame, (x_start, y_start),
                  (x_start + panel_w, y_start + panel_h),
                  (100, 100, 100), 1)

    # Title
    draw_label(frame, "INTELLITRAFFIC", (x_start + 5, y_start + 25),
               color=(0, 255, 255), bg_color=(30, 30, 30), font_scale=0.6, thickness=2)

    # Stats lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = y_start + 50
    line_height = 25

    stats_lines = [
        (f"Vehicles: {stats.get('vehicle_count', 0)}", (200, 200, 200)),
        (f"Avg Speed: {stats.get('avg_speed', 0):.1f} km/h", (200, 200, 200)),
        (f"Density: {stats.get('density_level', 'N/A')}", _get_density_color(stats.get('density_level', 'LOW'))),
        (f"Risk: {stats.get('risk_level', 'N/A')}", _get_risk_color(stats.get('risk_level', 'LOW'))),
        (f"Violations: {stats.get('violation_count', 0)}", (0, 100, 255) if stats.get('violation_count', 0) > 0 else (200, 200, 200)),
        (f"FPS: {stats.get('fps', 0):.1f}", (200, 200, 200)),
    ]

    for text, color in stats_lines:
        cv2.putText(frame, text, (x_start + 10, y_offset),
                    font, 0.5, color, 1, cv2.LINE_AA)
        y_offset += line_height

    return frame


def draw_speed_lines(frame, source_points: list, color: tuple = (0, 255, 255)):
    """
    Draw the speed estimation reference lines on the frame.

    Args:
        frame: Image to draw on.
        source_points: List of 4 [x,y] points defining the perspective region.
        color: Line color.
    """
    if not source_points or len(source_points) < 4:
        return frame

    pts = np.array(source_points, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

    # Draw corner circles
    for i, pt in enumerate(source_points):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
        cv2.putText(frame, f"P{i}", (int(pt[0]) + 8, int(pt[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def _get_density_color(level: str) -> tuple:
    """Map density level to display color."""
    mapping = {"LOW": COLORS["risk_low"], "MEDIUM": COLORS["risk_medium"], "HIGH": COLORS["risk_high"]}
    return mapping.get(level, (200, 200, 200))


def _get_risk_color(level: str) -> tuple:
    """Map risk level to display color."""
    mapping = {"LOW": COLORS["risk_low"], "MEDIUM": COLORS["risk_medium"], "HIGH": COLORS["risk_high"]}
    return mapping.get(level, (200, 200, 200))
