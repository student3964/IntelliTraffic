"""
============================================================
IntelliTraffic – Risk Engine
============================================================
Computes zone-level risk scores using weighted combination of
traffic density, average speed, and violation count.

Risk = w1 * density + w2 * avg_speed + w3 * violation_count
All values normalized to 0-1 range before weighting.
"""


class RiskEngine:
    """
    Computes and classifies zone-level traffic risk scores.
    """

    def __init__(self, config):
        """
        Initialize risk engine with configurable weights and thresholds.

        Args:
            config: Config object with risk engine settings.
        """
        self.enabled = config.get("risk", "enabled", default=True)

        # Weights for risk formula
        self.w_density = config.get("risk", "weights", "density", default=0.4)
        self.w_speed = config.get("risk", "weights", "avg_speed", default=0.3)
        self.w_violations = config.get("risk", "weights", "violation_count", default=0.3)

        # Risk level thresholds
        self.low_threshold = config.get("risk", "low_threshold", default=0.3)
        self.high_threshold = config.get("risk", "high_threshold", default=0.7)

        # Update interval (frames)
        self.update_interval = config.get("risk", "update_interval", default=30)

        # Normalization ranges
        self.max_density = config.get("risk", "max_density", default=30)
        self.max_speed = config.get("risk", "max_speed", default=120)
        self.max_violations = config.get("risk", "max_violations", default=10)

        # State
        self.current_score = 0.0
        self.current_level = "LOW"
        self.frame_counter = 0

        # History for trend analysis
        self.score_history = []
        self.max_history = 100

        print(f"[Risk] Initialized. Weights: density={self.w_density}, "
              f"speed={self.w_speed}, violations={self.w_violations}")

    def update(self, density_count: int, avg_speed: float,
               violation_count: int, force: bool = False) -> dict:
        """
        Update risk score based on current metrics.

        Args:
            density_count: Number of vehicles in the zone.
            avg_speed: Average speed of vehicles (km/h).
            violation_count: Number of active/recent violations.
            force: Force update regardless of interval.

        Returns:
            Dict with: score (0-1), level ("LOW"/"MEDIUM"/"HIGH"),
                        components (individual normalized values).
        """
        if not self.enabled:
            return {"score": 0.0, "level": "LOW", "components": {}}

        self.frame_counter += 1

        # Only update at specified intervals (or if forced)
        if not force and self.frame_counter % self.update_interval != 0:
            return self.get_current()

        # Normalize each component to 0-1 range
        norm_density = min(density_count / self.max_density, 1.0)
        norm_speed = min(avg_speed / self.max_speed, 1.0)
        norm_violations = min(violation_count / self.max_violations, 1.0)

        # Compute weighted risk score
        self.current_score = (
            self.w_density * norm_density +
            self.w_speed * norm_speed +
            self.w_violations * norm_violations
        )

        # Clamp to 0-1
        self.current_score = max(0.0, min(1.0, self.current_score))

        # Classify risk level
        if self.current_score < self.low_threshold:
            self.current_level = "LOW"
        elif self.current_score < self.high_threshold:
            self.current_level = "MEDIUM"
        else:
            self.current_level = "HIGH"

        # Store in history
        self.score_history.append(self.current_score)
        if len(self.score_history) > self.max_history:
            self.score_history = self.score_history[-self.max_history:]

        return {
            "score": round(self.current_score, 3),
            "level": self.current_level,
            "components": {
                "density": round(norm_density, 3),
                "speed": round(norm_speed, 3),
                "violations": round(norm_violations, 3)
            }
        }

    def get_current(self) -> dict:
        """Return the current risk assessment."""
        return {
            "score": round(self.current_score, 3),
            "level": self.current_level
        }

    def get_trend(self) -> str:
        """
        Determine if risk is trending up, down, or stable.

        Returns:
            "INCREASING", "DECREASING", or "STABLE"
        """
        if len(self.score_history) < 5:
            return "STABLE"

        recent = self.score_history[-5:]
        older = self.score_history[-10:-5] if len(self.score_history) >= 10 else self.score_history[:5]

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        diff = avg_recent - avg_older
        if diff > 0.05:
            return "INCREASING"
        elif diff < -0.05:
            return "DECREASING"
        return "STABLE"
