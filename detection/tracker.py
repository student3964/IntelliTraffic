"""
============================================================
IntelliTraffic – Multi-Object Tracking (DeepSORT)
============================================================
Wraps the deep-sort-realtime library to assign persistent
unique IDs to detected vehicles across frames.

Maintains position history per track for speed estimation.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict


class VehicleTracker:
    """
    DeepSORT-based multi-object tracker.

    Takes raw detections from YOLOv8 and returns tracked objects
    with persistent IDs and position history.
    """

    def __init__(self, config):
        """
        Initialize DeepSORT tracker.

        Args:
            config: Config object with tracker settings.
        """
        max_age = config.get("tracker", "max_age", default=30)
        n_init = config.get("tracker", "n_init", default=3)
        max_iou_distance = config.get("tracker", "max_iou_distance", default=0.7)
        embedder = config.get("tracker", "embedder", default="mobilenet")
        embedder_gpu = config.get("tracker", "embedder_gpu", default=False)

        # Initialize DeepSORT
        print(f"[Tracker] Initializing DeepSORT (embedder={embedder}, max_age={max_age})")
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu
        )

        # Store position history for each track: {track_id: [(cx, cy, timestamp), ...]}
        self.track_history = defaultdict(list)

        # Store class name mapping: {track_id: class_name}
        self.track_classes = {}

        # Maximum history length to keep per track
        self.max_history = 60

        print("[Tracker] DeepSORT initialized successfully")

    def update(self, detections: list, frame, frame_time: float = None) -> list:
        """
        Update tracker with new detections and return tracked objects.

        Args:
            detections: List of detection dicts from VehicleDetector.
                Each has: bbox [x1,y1,x2,y2], confidence, class_id, class_name
            frame: Current video frame (used for ReID feature extraction).
            frame_time: Current timestamp in seconds (for speed calculation).

        Returns:
            List of tracked object dicts:
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "class_name": str,
                    "confidence": float,
                    "is_confirmed": bool
                }
        """
        if not detections:
            # Still need to call update to age out old tracks
            tracks = self.tracker.update_tracks([], frame=frame)
            return self._format_tracks(tracks, frame_time)

        # Convert detections to DeepSORT format:
        # List of ([x1, y1, w, h], confidence, class_name)
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            deepsort_detections.append(
                ([x1, y1, w, h], det["confidence"], det["class_name"])
            )

        # Update DeepSORT tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)

        return self._format_tracks(tracks, frame_time)

    def _format_tracks(self, tracks, frame_time: float = None) -> list:
        """
        Format DeepSORT tracks into our standard dict format.

        Also updates position history for each confirmed track.

        Args:
            tracks: Raw DeepSORT track objects.
            frame_time: Current timestamp in seconds.

        Returns:
            List of tracked object dicts.
        """
        tracked_objects = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            # Get class name from detection info
            det_class = track.det_class if hasattr(track, 'det_class') and track.det_class else "vehicle"

            # Update class mapping (keep most recent)
            if det_class and det_class != "vehicle":
                self.track_classes[track_id] = det_class
            class_name = self.track_classes.get(track_id, det_class)

            # Get confidence
            det_conf = track.det_conf if hasattr(track, 'det_conf') and track.det_conf is not None else 0.0

            # Compute center point
            cx = (ltrb[0] + ltrb[2]) / 2
            cy = (ltrb[1] + ltrb[3]) / 2

            # Update position history with timestamp
            self.track_history[track_id].append((cx, cy, frame_time))

            # Trim history to max length
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id] = self.track_history[track_id][-self.max_history:]

            tracked_objects.append({
                "track_id": track_id,
                "bbox": [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                "class_name": class_name,
                "confidence": float(det_conf) if det_conf else 0.0,
                "is_confirmed": True
            })

        return tracked_objects

    def get_history(self, track_id: int) -> list:
        """
        Get the position history for a given track.

        Args:
            track_id: The unique track identifier.

        Returns:
            List of (cx, cy, timestamp) tuples, oldest first.
        """
        return self.track_history.get(track_id, [])

    def cleanup_old_tracks(self, active_ids: set):
        """
        Remove history for tracks that are no longer active.

        Args:
            active_ids: Set of currently active track IDs.
        """
        stale_ids = [tid for tid in self.track_history if tid not in active_ids]
        for tid in stale_ids:
            del self.track_history[tid]
            if tid in self.track_classes:
                del self.track_classes[tid]
