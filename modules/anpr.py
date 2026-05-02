"""
============================================================
IntelliTraffic – ANPR (Automatic Number Plate Recognition)
============================================================
Detects and reads vehicle number plates using EasyOCR.
Pipeline: crop vehicle region → preprocess → OCR → cleanup.
Caches results per track_id to avoid redundant OCR.
"""

import re
import cv2
import numpy as np


class PlateRecognizer:
    """
    Number plate recognition using EasyOCR on vehicle crops.
    """

    def __init__(self, config):
        self.enabled = config.get("anpr", "enabled", default=True)
        self.languages = config.get("anpr", "languages", default=["en"])
        self.gpu = config.get("anpr", "gpu", default=False)
        self.confidence_threshold = config.get("anpr", "confidence_threshold", default=0.3)
        self.plate_pattern = config.get("anpr", "plate_pattern", default=r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}")
        self.cache_results = config.get("anpr", "cache_results", default=True)

        # Cache: {track_id: plate_string}
        self.plate_cache = {}

        # Lazy-load EasyOCR reader (heavy import)
        self.reader = None

        if self.enabled:
            print(f"[ANPR] Initializing EasyOCR (languages={self.languages}, gpu={self.gpu})")
            try:
                import easyocr
                self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
                print("[ANPR] EasyOCR initialized successfully")
            except ImportError:
                print("[ANPR] WARNING: easyocr not installed. ANPR disabled.")
                self.enabled = False
            except Exception as e:
                print(f"[ANPR] WARNING: EasyOCR init failed: {e}. ANPR disabled.")
                self.enabled = False

    def recognize(self, frame: np.ndarray, tracked_objects: list) -> dict:
        """
        Attempt to read number plates for tracked vehicles.

        Args:
            frame: Full video frame (BGR).
            tracked_objects: List of tracked vehicle dicts.

        Returns:
            Dict mapping track_id → plate_string (or None).
        """
        if not self.enabled or self.reader is None:
            return {}

        results = {}
        for obj in tracked_objects:
            track_id = obj["track_id"]
            if obj.get("class_name") == "person":
                continue

            # Check cache first
            if self.cache_results and track_id in self.plate_cache:
                results[track_id] = self.plate_cache[track_id]
                continue

            # Crop vehicle region from frame
            x1, y1, x2, y2 = [int(c) for c in obj["bbox"]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 30 or y2 - y1 < 30:
                continue

            crop = frame[y1:y2, x1:x2]

            # Focus on lower portion of vehicle (where plate usually is)
            crop_h = crop.shape[0]
            plate_region = crop[int(crop_h * 0.5):, :]

            if plate_region.size == 0:
                continue

            # Preprocess for better OCR
            processed = self._preprocess(plate_region)

            # Run OCR
            plate_text = self._run_ocr(processed)

            if plate_text:
                results[track_id] = plate_text
                if self.cache_results:
                    self.plate_cache[track_id] = plate_text

        return results

    def recognize_single(self, frame: np.ndarray, bbox: list) -> str:
        """
        Recognize plate for a single vehicle (used for violation snapshots).

        Args:
            frame: Full video frame.
            bbox: [x1, y1, x2, y2] bounding box.

        Returns:
            Plate string or None.
        """
        if not self.enabled or self.reader is None:
            return None

        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 30 or y2 - y1 < 30:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_h = crop.shape[0]
        plate_region = crop[int(crop_h * 0.5):, :]

        if plate_region.size == 0:
            return None

        processed = self._preprocess(plate_region)
        return self._run_ocr(processed)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        Steps: grayscale → bilateral filter → adaptive threshold.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Bilateral filter reduces noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        # Adaptive threshold for varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def _run_ocr(self, image: np.ndarray) -> str:
        """
        Run EasyOCR on preprocessed image and clean results.

        Returns:
            Cleaned plate string or None.
        """
        try:
            results = self.reader.readtext(image)
        except Exception:
            return None

        if not results:
            return None

        # Combine all detected text segments
        all_text = ""
        for (bbox, text, confidence) in results:
            if confidence >= self.confidence_threshold:
                all_text += text

        if not all_text:
            return None

        # Clean and format plate text
        cleaned = self._clean_plate(all_text)
        return cleaned if cleaned else None

    def _clean_plate(self, raw_text: str) -> str:
        """
        Clean OCR output to extract valid plate number.

        Steps:
            1. Remove special characters and spaces
            2. Convert to uppercase
            3. Try to match plate pattern
            4. Apply common OCR corrections (0→O, 1→I, etc.)
        """
        # Remove everything except alphanumeric
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

        if not cleaned:
            return None

        # Common OCR character corrections
        corrections = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6'}
        # Only apply to digit positions (not letter positions)

        # Try to match the expected plate pattern
        match = re.search(self.plate_pattern, cleaned)
        if match:
            return match.group()

        # If no pattern match, return cleaned text if it looks reasonable
        if len(cleaned) >= 4:
            return cleaned

        return None

    def get_plate(self, track_id: int) -> str:
        """Get cached plate for a track_id."""
        return self.plate_cache.get(track_id)

    def clear_cache(self, track_ids: set = None):
        """Clear plate cache for specific tracks or all."""
        if track_ids is None:
            self.plate_cache.clear()
        else:
            for tid in track_ids:
                self.plate_cache.pop(tid, None)
