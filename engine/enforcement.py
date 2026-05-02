"""
============================================================
IntelliTraffic – E-Challan & Enforcement Engine
============================================================
Generates challan records for detected violations, stores them
in SQLite and CSV, captures violation snapshots, and optionally
sends email notifications via SMTP.
"""

import os
import csv
import time
import sqlite3
import smtplib
import cv2
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime


class EnforcementEngine:
    """
    Handles e-challan generation, storage, and notifications.
    """

    def __init__(self, config):
        """
        Initialize enforcement engine with DB, CSV, and email settings.
        """
        self.enabled = config.get("enforcement", "enabled", default=True)

        # Database paths
        self.db_path = config.get("enforcement", "db_path", default="data/challans.db")
        self.csv_path = config.get("enforcement", "csv_path", default="data/logs.csv")
        self.snapshot_dir = config.get("enforcement", "snapshot_dir", default="data/snapshots")

        # Email settings
        self.email_enabled = config.get("enforcement", "email", "enabled", default=False)
        self.smtp_host = config.get("enforcement", "email", "smtp_host", default="smtp.gmail.com")
        self.smtp_port = config.get("enforcement", "email", "smtp_port", default=465)
        self.sender_email = config.get("enforcement", "email", "sender_email", default="")
        self.sender_password = config.get("enforcement", "email", "sender_password", default="")
        self.recipient_email = config.get("enforcement", "email", "recipient_email", default="")
        self.subject_prefix = config.get("enforcement", "email", "subject_prefix",
                                         default="[IntelliTraffic] E-Challan")

        # Challan counter
        self.challan_count = 0

        if self.enabled:
            self._init_storage()
            print(f"[Enforcement] Initialized. DB: {self.db_path}, CSV: {self.csv_path}")
            if self.email_enabled:
                print(f"[Enforcement] Email notifications enabled via {self.smtp_host}")

    def _init_storage(self):
        """Create database, CSV, and snapshot directories."""
        # Create directories
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS challans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challan_id TEXT UNIQUE,
                vehicle_id INTEGER,
                plate_number TEXT,
                violation_type TEXT,
                details TEXT,
                speed REAL,
                timestamp TEXT,
                snapshot_path TEXT,
                class_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        # Initialize CSV with headers if new file
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "challan_id", "vehicle_id", "plate_number",
                    "violation_type", "details", "speed",
                    "timestamp", "snapshot_path", "class_name"
                ])

    def process_violation(self, violation: dict, frame: np.ndarray = None,
                          plate_number: str = None) -> dict:
        """
        Process a violation: create challan, save snapshot, store to DB/CSV.

        Args:
            violation: Violation dict from ViolationDetector.
            frame: Current video frame (for snapshot capture).
            plate_number: Recognized plate number (from ANPR).

        Returns:
            Challan record dict.
        """
        if not self.enabled:
            return {}

        self.challan_count += 1
        timestamp = datetime.fromtimestamp(violation["timestamp"])
        challan_id = f"IT-{timestamp.strftime('%Y%m%d')}-{self.challan_count:04d}"

        # Save violation snapshot
        snapshot_path = None
        if frame is not None:
            snapshot_path = self._save_snapshot(frame, violation, challan_id)

        # Build challan record
        challan = {
            "challan_id": challan_id,
            "vehicle_id": violation["track_id"],
            "plate_number": plate_number or "UNKNOWN",
            "violation_type": violation["violation_type"],
            "details": violation.get("details", ""),
            "speed": violation.get("speed"),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_path": snapshot_path or "",
            "class_name": violation.get("class_name", "vehicle")
        }

        # Store to SQLite
        self._save_to_db(challan)

        # Append to CSV
        self._save_to_csv(challan)

        # Send email notification (non-blocking, with error handling)
        if self.email_enabled:
            try:
                self._send_email(challan, snapshot_path)
            except Exception as e:
                print(f"[Enforcement] Email failed: {e}")

        print(f"[Enforcement] Challan {challan_id}: {violation['violation_type']} "
              f"| Vehicle #{violation['track_id']} | Plate: {challan['plate_number']}")

        return challan

    def _save_snapshot(self, frame: np.ndarray, violation: dict,
                       challan_id: str) -> str:
        """
        Save a cropped violation snapshot with annotation.

        Args:
            frame: Full video frame.
            violation: Violation dict containing bbox.
            challan_id: Challan identifier for filename.

        Returns:
            Path to saved snapshot.
        """
        snapshot = frame.copy()

        # Draw violation box on snapshot
        if "bbox" in violation:
            x1, y1, x2, y2 = [int(c) for c in violation["bbox"]]
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{violation['violation_type']} | {challan_id}"
            cv2.putText(snapshot, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add timestamp overlay
        ts_text = datetime.fromtimestamp(violation["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(snapshot, ts_text, (10, snapshot.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save snapshot
        filename = f"{challan_id}.jpg"
        filepath = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(filepath, snapshot)

        return filepath

    def _save_to_db(self, challan: dict):
        """Insert challan record into SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO challans (challan_id, vehicle_id, plate_number,
                    violation_type, details, speed, timestamp, snapshot_path, class_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                challan["challan_id"], challan["vehicle_id"],
                challan["plate_number"], challan["violation_type"],
                challan["details"], challan["speed"],
                challan["timestamp"], challan["snapshot_path"],
                challan["class_name"]
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Enforcement] DB write error: {e}")

    def _save_to_csv(self, challan: dict):
        """Append challan record to CSV file."""
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    challan["challan_id"], challan["vehicle_id"],
                    challan["plate_number"], challan["violation_type"],
                    challan["details"], challan["speed"],
                    challan["timestamp"], challan["snapshot_path"],
                    challan["class_name"]
                ])
        except Exception as e:
            print(f"[Enforcement] CSV write error: {e}")

    def _send_email(self, challan: dict, snapshot_path: str = None):
        """
        Send e-challan notification via SMTP email.

        Args:
            challan: Challan record dict.
            snapshot_path: Optional path to violation snapshot.
        """
        if not self.sender_email or not self.recipient_email:
            return

        # Build HTML email
        msg = MIMEMultipart("related")
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email
        msg["Subject"] = f"{self.subject_prefix} - {challan['challan_id']}"

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #d32f2f;">🚨 Traffic Violation E-Challan</h2>
            <hr>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 8px; font-weight: bold;">Challan ID:</td>
                    <td style="padding: 8px;">{challan['challan_id']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Vehicle ID:</td>
                    <td style="padding: 8px;">#{challan['vehicle_id']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Plate Number:</td>
                    <td style="padding: 8px;">{challan['plate_number']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Violation:</td>
                    <td style="padding: 8px; color: #d32f2f;">{challan['violation_type']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Details:</td>
                    <td style="padding: 8px;">{challan['details']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Speed:</td>
                    <td style="padding: 8px;">{challan['speed'] or 'N/A'} km/h</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                    <td style="padding: 8px;">{challan['timestamp']}</td></tr>
            </table>
            <hr>
            <p style="color: #666; font-size: 12px;">
                This is an automated notification from IntelliTraffic System.
            </p>
        </body>
        </html>
        """
        msg.attach(MIMEText(html, "html"))

        # Attach snapshot if available
        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, "rb") as img_f:
                img_data = MIMEImage(img_f.read(), name=os.path.basename(snapshot_path))
                msg.attach(img_data)

        # Send via SMTP
        with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as server:
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)

        print(f"[Enforcement] Email sent for challan {challan['challan_id']}")

    def get_all_challans(self) -> list:
        """Retrieve all challans from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM challans ORDER BY id DESC")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            return [dict(zip(columns, row)) for row in rows]
        except Exception:
            return []

    def get_challan_count(self) -> int:
        """Get total number of challans issued."""
        return self.challan_count
