"""
============================================================
IntelliTraffic – Traffic Density Estimation Module
============================================================
Counts vehicles within a defined Region of Interest (ROI),
computes density classification, and generates a heatmap
visualization of vehicle positions over time.
"""

import cv2
import numpy as np


class DensityAnalyzer:
    """
    Analyzes traffic density within a configurable ROI polygon.
    Maintains a cumulative heatmap for visualization.
    """

    def __init__(self, config):
        """
        Initialize density analyzer.

        Args:
            config: Config object with density settings.
        """
        self.enabled = config.get("density", "enabled", default=True)

        # ROI polygon vertices
        roi_pts = config.get("density", "roi_points",
                             default=[[100, 300], [1180, 300], [1200, 700], [80, 700]])
        self.roi_points = np.array(roi_pts, dtype=np.int32)

        # Density thresholds
        self.low_threshold = config.get("density", "low_threshold", default=5)
        self.high_threshold = config.get("density", "high_threshold", default=15)

        # Heatmap settings
        self.heatmap_enabled = config.get("density", "heatmap_enabled", default=True)
        self.heatmap_decay = config.get("density", "heatmap_decay", default=0.95)
        self.heatmap_intensity = config.get("density", "heatmap_intensity", default=5)

        # State variables
        self.current_count = 0
        self.density_level = "LOW"
        self.heatmap = None  # Initialized on first frame

        print(f"[Density] Initialized. Thresholds: LOW<{self.low_threshold}, "
              f"HIGH>{self.high_threshold}")

    def _is_inside_roi(self, point: tuple) -> bool:
        """
        Check if a point is inside the ROI polygon.

        Args:
            point: (x, y) coordinates.

        Returns:
            True if point is inside the ROI.
        """
        result = cv2.pointPolygonTest(self.roi_points, point, False)
        return result >= 0  # >= 0 means inside or on edge

    def analyze(self, tracked_objects: list, frame_shape: tuple = None) -> dict:
        """
        Count vehicles in ROI and classify density.

        Args:
            tracked_objects: List of tracked vehicle dicts with bbox.
            frame_shape: (height, width) of the video frame for heatmap init.

        Returns:
            Dict with:
                count: number of vehicles in ROI
                level: "LOW", "MEDIUM", or "HIGH"
                vehicles_in_roi: list of track_ids inside ROI
        """
        if not self.enabled:
            return {"count": 0, "level": "LOW", "vehicles_in_roi": []}

        # Initialize heatmap if needed
        if self.heatmap is None and frame_shape is not None:
            self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        vehicles_in_roi = []

        for obj in tracked_objects:
            # Skip person detections (only count vehicles)
            if obj.get("class_name") == "person":
                continue

            # Compute center of bounding box
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Check if vehicle center is inside ROI
            if self._is_inside_roi((cx, cy)):
                vehicles_in_roi.append(obj["track_id"])

                # Update heatmap (add heat at vehicle position)
                if self.heatmap_enabled and self.heatmap is not None:
                    ix, iy = int(cx), int(cy)
                    if 0 <= iy < self.heatmap.shape[0] and 0 <= ix < self.heatmap.shape[1]:
                        # Add a Gaussian blob at the vehicle center
                        self._add_heatmap_point(ix, iy)

        # Update density count and level
        self.current_count = len(vehicles_in_roi)
        self.density_level = self._classify_density(self.current_count)

        # Apply decay to heatmap (old positions fade)
        if self.heatmap_enabled and self.heatmap is not None:
            self.heatmap *= self.heatmap_decay

        return {
            "count": self.current_count,
            "level": self.density_level,
            "vehicles_in_roi": vehicles_in_roi
        }

    def _add_heatmap_point(self, x: int, y: int, radius: int = 30):
        """
        Add a Gaussian heat blob at the given position.

        Args:
            x, y: Center position.
            radius: Radius of the Gaussian blob.
        """
        h, w = self.heatmap.shape
        # Create coordinate grids for the blob region
        y_min = max(0, y - radius)
        y_max = min(h, y + radius)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius)

        # Generate Gaussian values
        for iy in range(y_min, y_max):
            for ix in range(x_min, x_max):
                dist = np.sqrt((ix - x)**2 + (iy - y)**2)
                if dist < radius:
                    # Gaussian falloff
                    value = self.heatmap_intensity * np.exp(-0.5 * (dist / (radius / 3))**2)
                    self.heatmap[iy, ix] += value

    def _classify_density(self, count: int) -> str:
        """
        Classify density level based on vehicle count.

        Args:
            count: Number of vehicles in ROI.

        Returns:
            "LOW", "MEDIUM", or "HIGH"
        """
        if count <= self.low_threshold:
            return "LOW"
        elif count <= self.high_threshold:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_heatmap(self) -> np.ndarray:
        """Return the current heatmap array (or None if disabled)."""
        return self.heatmap

    def get_roi_points(self) -> np.ndarray:
        """Return ROI polygon points for visualization."""
        return self.roi_points

    def get_stats(self) -> dict:
        """Return current density statistics."""
        return {
            "count": self.current_count,
            "level": self.density_level
        }
