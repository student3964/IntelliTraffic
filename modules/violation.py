"""
============================================================
IntelliTraffic – Violation Detection Module
============================================================
Detects traffic violations:
    1. Overspeeding – vehicle exceeds speed limit
    2. Triple Riding – more than 2 persons on a motorcycle
    3. (Optional) Wrong Direction – vehicle moving against flow
"""

import time
from collections import defaultdict


class ViolationDetector:
    """Detects traffic violations from tracked vehicle data."""

    def __init__(self, config):
        self.overspeed_enabled = config.get("violations", "overspeeding", "enabled", default=True)
        self.speed_threshold = config.get("violations", "overspeeding", "threshold", default=60)
        self.triple_enabled = config.get("violations", "triple_riding", "enabled", default=True)
        self.iou_threshold = config.get("violations", "triple_riding", "iou_threshold", default=0.3)
        self.min_persons = config.get("violations", "triple_riding", "min_persons", default=3)
        self.wrong_dir_enabled = config.get("violations", "wrong_direction", "enabled", default=False)
        self.expected_direction = config.get("violations", "wrong_direction", "expected_direction", default="down")
        self.cooldown_seconds = config.get("violations", "cooldown_seconds", default=10)
        self.cooldown_tracker = {}
        self.all_violations = []
        print(f"[Violations] Overspeed: {self.overspeed_enabled} (>{self.speed_threshold}km/h), "
              f"Triple: {self.triple_enabled}")

    def detect(self, tracked_objects, persons, speeds, track_history=None):
        """Detect all violations in the current frame."""
        current_violations = []
        current_time = time.time()

        # 1. Overspeeding
        if self.overspeed_enabled:
            for obj in tracked_objects:
                tid = obj["track_id"]
                if obj.get("class_name") == "person":
                    continue
                speed = speeds.get(tid, 0.0)
                if speed > self.speed_threshold:
                    if self._check_cooldown(tid, "OVERSPEEDING", current_time):
                        v = {"track_id": tid, "violation_type": "OVERSPEEDING",
                             "details": f"{speed:.1f} km/h (limit: {self.speed_threshold})",
                             "bbox": obj["bbox"], "speed": speed,
                             "timestamp": current_time, "class_name": obj.get("class_name", "vehicle")}
                        current_violations.append(v)
                        self.all_violations.append(v)

        # 2. Triple Riding
        if self.triple_enabled and persons:
            motos = [o for o in tracked_objects if o.get("class_name") == "motorcycle"]
            for moto in motos:
                tid = moto["track_id"]
                count = sum(1 for p in persons if self._compute_iou(moto["bbox"], p["bbox"]) > self.iou_threshold)
                if count >= self.min_persons:
                    if self._check_cooldown(tid, "TRIPLE_RIDING", current_time):
                        v = {"track_id": tid, "violation_type": "TRIPLE_RIDING",
                             "details": f"{count} persons on motorcycle", "bbox": moto["bbox"],
                             "speed": speeds.get(tid), "timestamp": current_time, "class_name": "motorcycle"}
                        current_violations.append(v)
                        self.all_violations.append(v)

        # 3. Wrong Direction (Optional)
        if self.wrong_dir_enabled and track_history:
            for obj in tracked_objects:
                if obj.get("class_name") == "person":
                    continue
                tid = obj["track_id"]
                history = track_history.get(tid, [])
                if len(history) >= 5:
                    direction = self._detect_direction(history)
                    if direction and direction != self.expected_direction:
                        if self._check_cooldown(tid, "WRONG_DIRECTION", current_time):
                            v = {"track_id": tid, "violation_type": "WRONG_DIRECTION",
                                 "details": f"Moving {direction} (expected: {self.expected_direction})",
                                 "bbox": obj["bbox"], "speed": speeds.get(tid),
                                 "timestamp": current_time, "class_name": obj.get("class_name", "vehicle")}
                            current_violations.append(v)
                            self.all_violations.append(v)

        return current_violations

    def _check_cooldown(self, track_id, violation_type, current_time):
        key = (track_id, violation_type)
        last = self.cooldown_tracker.get(key, 0)
        if current_time - last > self.cooldown_seconds:
            self.cooldown_tracker[key] = current_time
            return True
        return False

    @staticmethod
    def _compute_iou(box1, box2):
        """Compute overlap ratio of box2 inside box1 (for person-on-motorcycle)."""
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0:
            return 0.0
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / area2 if area2 > 0 else 0.0

    @staticmethod
    def _detect_direction(history):
        if len(history) < 2:
            return None
        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        if abs(dy) > abs(dx):
            return "down" if dy > 0 else "up"
        return "right" if dx > 0 else "left"

    def get_violation_count(self):
        return len(self.all_violations)

    def get_recent_violations(self, n=10):
        return self.all_violations[-n:]

    def get_stats(self):
        type_counts = defaultdict(int)
        for v in self.all_violations:
            type_counts[v["violation_type"]] += 1
        return {"total": len(self.all_violations), "by_type": dict(type_counts)}
