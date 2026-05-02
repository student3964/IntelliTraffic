"""
============================================================
IntelliTraffic – Streamlit Dashboard
============================================================
Real-time traffic monitoring dashboard with:
    - Live video feed with detection overlays
    - Speed, density, and risk metrics
    - Violation log table
    - Challan history from SQLite
    - Heatmap toggle & UI controls

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import time
import cv2
import numpy as np
import streamlit as st
import sqlite3
import csv
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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


# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="IntelliTraffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Dark Premium Theme ──────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
    }

    /* Status indicators */
    .status-low { color: #48bb78; font-weight: bold; }
    .status-medium { color: #ecc94b; font-weight: bold; }
    .status-high { color: #fc8181; font-weight: bold; }

    /* Violation alert */
    .violation-alert {
        background: rgba(252, 129, 129, 0.15);
        border-left: 4px solid #fc8181;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #fed7d7;
    }

    /* Title bar */
    .title-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    .title-bar h1 {
        margin: 0 !important;
        font-size: 1.8rem !important;
        color: white !important;
    }
    .title-bar p {
        color: rgba(255,255,255,0.8);
        margin: 4px 0 0 0;
        font-size: 0.95rem;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1rem;
    }
    .risk-low { background: rgba(72, 187, 120, 0.2); color: #48bb78; border: 1px solid #48bb78; }
    .risk-medium { background: rgba(236, 201, 75, 0.2); color: #ecc94b; border: 1px solid #ecc94b; }
    .risk-high { background: rgba(252, 129, 129, 0.2); color: #fc8181; border: 1px solid #fc8181; }

    /* Data table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def get_status_html(level):
    """Generate colored status badge HTML."""
    css_class = f"risk-{level.lower()}"
    return f'<span class="risk-badge {css_class}">{level}</span>'


@st.cache_resource
def load_modules(config_path):
    """Load all pipeline modules (cached to avoid reloading)."""
    Config.reset()
    config = Config(config_path)
    return {
        "config": config,
        "detector": VehicleDetector(config),
        "tracker": VehicleTracker(config),
        "speed": SpeedEstimator(config),
        "density": DensityAnalyzer(config),
        "violations": ViolationDetector(config),
        "anpr": PlateRecognizer(config),
        "risk": RiskEngine(config),
        "enforcement": EnforcementEngine(config),
    }


def load_challans(db_path):
    """Load challan records from SQLite."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT challan_id, vehicle_id, plate_number, violation_type, "
                       "details, speed, timestamp FROM challans ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def main():
    """Main dashboard application."""

    # ── Title Bar ──────────────────────────────────────────
    st.markdown("""
    <div class="title-bar">
        <h1>🚦 IntelliTraffic Dashboard</h1>
        <p>Real-time Traffic Intelligence & Automated Enforcement System</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ───────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown("---")

        # Config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )

        # Video source
        source_type = st.radio("📹 Video Source", ["Upload File", "Camera (Webcam)", "File Path"])

        video_source = None
        uploaded_file = None

        if source_type == "Upload File":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file:
                # Save uploaded file temporarily
                temp_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "temp_upload.mp4"
                )
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                video_source = temp_path
        elif source_type == "Camera (Webcam)":
            video_source = 0
        else:
            video_source = st.text_input("Video File Path", value="sample.mp4")

        st.markdown("---")
        st.markdown("### 🎛️ Parameters")

        speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 60, 5)
        confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.4, 0.05)
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        show_roi = st.checkbox("Show ROI Region", value=True)
        show_speed_lines = st.checkbox("Show Speed Lines", value=False)

        st.markdown("---")

        # Start/Stop controls
        start_btn = st.button("▶️ Start Processing", type="primary", use_container_width=True)
        stop_btn = st.button("⏹️ Stop", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")

    # ── Main Content Area ──────────────────────────────────

    if start_btn and video_source is not None:
        # Load modules
        with st.spinner("🔄 Loading AI models..."):
            modules = load_modules(config_path)

        # Override speed limit from slider
        modules["speed"].speed_limit = speed_limit
        modules["violations"].speed_threshold = speed_limit
        modules["detector"].confidence = confidence

        # Open video
        try:
            if isinstance(video_source, int):
                cap = cv2.VideoCapture(video_source)
            else:
                cap = cv2.VideoCapture(str(video_source))

            if not cap.isOpened():
                st.error(f"❌ Cannot open video source: {video_source}")
                return
        except Exception as e:
            st.error(f"❌ Error opening video: {e}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_delta = 1.0 / fps

        # Layout: video + metrics
        col_video, col_metrics = st.columns([3, 1])

        with col_video:
            st.markdown("### 📹 Live Feed")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)

        with col_metrics:
            st.markdown("### 📈 Real-time Metrics")
            metric_vehicle = st.empty()
            metric_speed = st.empty()
            metric_density = st.empty()
            metric_risk = st.empty()
            metric_violations = st.empty()
            metric_fps = st.empty()

        # Violations section
        st.markdown("---")
        col_violations, col_challans = st.columns(2)

        with col_violations:
            st.markdown("### 🚨 Recent Violations")
            violations_placeholder = st.empty()

        with col_challans:
            st.markdown("### 📋 E-Challan Log")
            challans_placeholder = st.empty()

        # Processing loop
        frame_count = 0
        loop_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count * frame_delta

            # Resize for performance
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # --- Run pipeline ---
            detections = modules["detector"].detect(frame)
            tracked = modules["tracker"].update(
                detections["vehicles"] + detections["persons"],
                frame, frame_time=current_time
            )
            speeds = modules["speed"].estimate_speeds(tracked, modules["tracker"].track_history)
            density = modules["density"].analyze(tracked, frame.shape[:2])
            violations = modules["violations"].detect(
                tracked, detections["persons"], speeds, modules["tracker"].track_history
            )

            # ANPR on violations
            if violations:
                v_objs = [o for o in tracked if o["track_id"] in {v["track_id"] for v in violations}]
                plates = modules["anpr"].recognize(frame, v_objs)
                for v in violations:
                    modules["enforcement"].process_violation(
                        v, frame, plates.get(v["track_id"])
                    )

            risk = modules["risk"].update(
                density["count"],
                modules["speed"].get_average_speed(),
                modules["violations"].get_violation_count()
            )

            # --- Draw frame ---
            display = frame.copy()
            if show_roi:
                draw_roi(display, modules["density"].get_roi_points().tolist())
            if show_speed_lines:
                draw_speed_lines(display, modules["speed"].get_source_points())
            draw_tracked_vehicles(display, tracked, speeds)
            draw_violations(display, violations)
            if show_heatmap:
                hm = modules["density"].get_heatmap()
                if hm is not None:
                    display = draw_density_heatmap(display, hm, alpha=0.3)

            # Convert BGR→RGB for Streamlit
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            # Update UI (throttled to avoid overwhelming Streamlit)
            if frame_count % 2 == 0:
                video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

                elapsed = time.time() - loop_start
                cur_fps = frame_count / elapsed if elapsed > 0 else 0

                metric_vehicle.metric("🚗 Vehicles in Zone", density["count"])
                metric_speed.metric("⚡ Avg Speed", f"{modules['speed'].get_average_speed():.1f} km/h")
                metric_density.metric("📊 Density", density["level"])
                metric_risk.metric("⚠️ Risk Level", risk.get("level", "LOW"))
                metric_violations.metric("🚨 Violations", modules["violations"].get_violation_count())
                metric_fps.metric("🎯 FPS", f"{cur_fps:.1f}")

                # Update progress
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                # Update violations table
                recent = modules["violations"].get_recent_violations(10)
                if recent:
                    import pandas as pd
                    v_data = [{
                        "Type": v["violation_type"],
                        "Vehicle": f"#{v['track_id']}",
                        "Details": v["details"],
                        "Time": datetime.fromtimestamp(v["timestamp"]).strftime("%H:%M:%S")
                    } for v in reversed(recent)]
                    violations_placeholder.dataframe(pd.DataFrame(v_data), use_container_width=True)

                # Update challans table
                db_path = modules["config"].get("enforcement", "db_path", default="data/challans.db")
                challan_rows = load_challans(db_path)
                if challan_rows:
                    import pandas as pd
                    c_data = [{
                        "Challan ID": r[0], "Vehicle": f"#{r[1]}",
                        "Plate": r[2], "Violation": r[3],
                        "Speed": f"{r[5]} km/h" if r[5] else "N/A",
                        "Time": r[6]
                    } for r in challan_rows[:10]]
                    challans_placeholder.dataframe(pd.DataFrame(c_data), use_container_width=True)

            # Check for stop (Streamlit reruns on interaction)
            if stop_btn:
                break

        cap.release()
        progress_bar.progress(1.0)
        st.success(f"✅ Processing complete! {frame_count} frames processed.")

        # Final summary
        st.markdown("---")
        st.markdown("### 📊 Session Summary")
        total_time = time.time() - loop_start

        sum_cols = st.columns(4)
        sum_cols[0].metric("Total Frames", frame_count)
        sum_cols[1].metric("Processing Time", f"{total_time:.1f}s")
        sum_cols[2].metric("Avg FPS", f"{frame_count / total_time:.1f}" if total_time > 0 else "0")
        sum_cols[3].metric("Challans Issued", modules["enforcement"].get_challan_count())

        # Download challans CSV
        csv_path = modules["config"].get("enforcement", "csv_path", default="data/logs.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                st.download_button(
                    "📥 Download Challan Logs (CSV)",
                    f.read(),
                    file_name="intellitraffic_challans.csv",
                    mime="text/csv"
                )

    else:
        # Landing state - show instructions and existing data
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #a0aec0;">
            <h2 style="color: #667eea !important;">Welcome to IntelliTraffic</h2>
            <p style="font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Upload a traffic video or connect a webcam to begin real-time
                traffic analysis, violation detection, and automated enforcement.
            </p>
            <br>
            <p>👈 Use the sidebar to configure and start processing</p>
        </div>
        """, unsafe_allow_html=True)

        # Show existing challan data if available
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "challans.db"
        )
        challan_rows = load_challans(db_path)
        if challan_rows:
            st.markdown("---")
            st.markdown("### 📋 Previous Session – Challan History")
            import pandas as pd
            c_data = [{
                "Challan ID": r[0], "Vehicle": f"#{r[1]}",
                "Plate": r[2], "Violation": r[3],
                "Details": r[4], "Speed": f"{r[5]} km/h" if r[5] else "N/A",
                "Time": r[6]
            } for r in challan_rows]
            st.dataframe(pd.DataFrame(c_data), use_container_width=True)


if __name__ == "__main__":
    main()
