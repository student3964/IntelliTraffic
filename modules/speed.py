"""
============================================================
IntelliTraffic – Speed Estimation Module
============================================================
Estimates vehicle speed using perspective transform to convert
pixel displacement to real-world distance, then calculates
speed from distance/time.

Approach:
    1. Define 4 source points in the video (road surface)
    2. Define 4 destination points in real-world meters
    3. Use cv2.getPerspectiveTransform to build the mapping
    4. Transform tracked positions to real-world coordinates
    5. Calculate speed = distance / time * 3.6 (m/s → km/h)
    6. Apply rolling average for smoothing
"""

import cv2
import numpy as np
from collections import defaultdict


class SpeedEstimator:
    """
    Estimates speed for tracked vehicles using perspective transformation.
    """

    def __init__(self, config):
        """
        Initialize speed estimator with perspective calibration.

        Args:
            config: Config object with speed estimation settings.
        """
        self.enabled = config.get("speed", "enabled", default=True)
        self.speed_limit = config.get("speed", "speed_limit", default=60)
        self.smoothing_window = config.get("speed", "smoothing_window", default=5)
        self.min_track_length = config.get("speed", "min_track_length", default=3)

        # Perspective transform calibration points
        src_pts = config.get("speed", "source_points",
                             default=[[300, 400], [980, 400], [1200, 700], [100, 700]])
        dst_pts = config.get("speed", "dest_points",
                             default=[[0, 0], [10, 0], [10, 30], [0, 30]])

        self.source_points = np.float32(src_pts)
        self.dest_points = np.float32(dst_pts)

        # Compute the perspective transformation matrix
        self.transform_matrix = cv2.getPerspectiveTransform(
            self.source_points, self.dest_points
        )

        # Store speed readings per track for smoothing: {track_id: [speed1, speed2, ...]}
        self.speed_buffer = defaultdict(list)

        # Store the latest computed speed per track: {track_id: smoothed_speed}
        self.current_speeds = {}

        print(f"[Speed] Initialized. Limit: {self.speed_limit} km/h, "
              f"Smoothing window: {self.smoothing_window}")

    def _pixel_to_world(self, pixel_point: tuple) -> tuple:
        """
        Convert a pixel coordinate to real-world coordinate (meters)
        using the perspective transform.

        Args:
            pixel_point: (x, y) in pixel space.

        Returns:
            (x, y) in real-world meters.
        """
        pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.transform_matrix)
        return (transformed[0][0][0], transformed[0][0][1])

    def estimate_speeds(self, tracked_objects: list, track_history: dict) -> dict:
        """
        Estimate speed for all tracked vehicles.

        Uses the vehicle's position history from the tracker to compute
        displacement in real-world coordinates over time.

        Args:
            tracked_objects: List of tracked vehicle dicts with track_id.
            track_history: Dict from tracker: {track_id: [(cx, cy, timestamp), ...]}.

        Returns:
            Dict mapping track_id → speed in km/h.
        """
        if not self.enabled:
            return {}

        for obj in tracked_objects:
            track_id = obj["track_id"]
            history = track_history.get(track_id, [])

            if len(history) < self.min_track_length:
                continue

            # Get the two most recent positions with timestamps
            # Use a gap of a few frames for more stable measurement
            gap = min(self.smoothing_window, len(history) - 1)
            recent = history[-1]       # (cx, cy, timestamp)
            earlier = history[-1 - gap]  # Earlier position

            # Skip if timestamps are invalid
            if recent[2] is None or earlier[2] is None:
                continue

            time_diff = recent[2] - earlier[2]
            if time_diff <= 0:
                continue

            # Convert pixel positions to real-world coordinates
            try:
                world_recent = self._pixel_to_world((recent[0], recent[1]))
                world_earlier = self._pixel_to_world((earlier[0], earlier[1]))
            except Exception:
                continue

            # Calculate Euclidean distance in real-world meters
            dx = world_recent[0] - world_earlier[0]
            dy = world_recent[1] - world_earlier[1]
            distance_meters = np.sqrt(dx**2 + dy**2)

            # Speed in m/s → km/h
            speed_mps = distance_meters / time_diff
            speed_kmh = speed_mps * 3.6

            # Sanity check: cap at reasonable values
            speed_kmh = min(speed_kmh, 200.0)
            speed_kmh = max(speed_kmh, 0.0)

            # Add to smoothing buffer
            self.speed_buffer[track_id].append(speed_kmh)

            # Keep buffer limited
            if len(self.speed_buffer[track_id]) > self.smoothing_window * 2:
                self.speed_buffer[track_id] = self.speed_buffer[track_id][-self.smoothing_window * 2:]

            # Compute rolling average (smoothed speed)
            buffer = self.speed_buffer[track_id][-self.smoothing_window:]
            smoothed_speed = sum(buffer) / len(buffer)

            self.current_speeds[track_id] = round(smoothed_speed, 1)

        # Clean up old tracks
        active_ids = {obj["track_id"] for obj in tracked_objects}
        stale = [tid for tid in self.current_speeds if tid not in active_ids]
        for tid in stale:
            del self.current_speeds[tid]
            if tid in self.speed_buffer:
                del self.speed_buffer[tid]

        return self.current_speeds.copy()

    def get_speed(self, track_id: int) -> float:
        """Get the current speed for a specific track."""
        return self.current_speeds.get(track_id, 0.0)

    def get_average_speed(self) -> float:
        """Get the average speed of all currently tracked vehicles."""
        if not self.current_speeds:
            return 0.0
        return sum(self.current_speeds.values()) / len(self.current_speeds)

    def is_overspeeding(self, track_id: int) -> bool:
        """Check if a specific vehicle is exceeding the speed limit."""
        return self.current_speeds.get(track_id, 0.0) > self.speed_limit

    def get_source_points(self) -> list:
        """Return the perspective source points for visualization."""
        return self.source_points.tolist()
