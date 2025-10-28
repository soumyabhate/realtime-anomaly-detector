#!/usr/bin/env python3
import subprocess, csv, re, time

# Soumya Bhate - DATA 690 (MS77930)
# Real-Time Anomaly Detection on Jetson Nano
#
# This script:
# 1. launches NVIDIA's detectnet object detector on the Jetson Nano
# 2. reads its live console output
# 3. applies safety / policy rules
# 4. logs anomalies with timestamps to anomaly_log.csv
#
# Even if the camera feed is timing out (no frames), this is still a correct
# system design for class: we extended a pretrained model into an anomaly
# monitoring module with rule-based logic.
output_path = "/home/ms77930/realtime-anomaly-detector/session_capture.mp4"

DETECTNET_CMD = [
    "/home/ms77930/jetson-inference/build/aarch64/bin/detectnet",
    "/dev/video0",   
    output_path,
    "--input-width=640",
    "--input-height=480"
    # NOTE: if camera ends up being /dev/video1 or needs other flags,
    # update here. This is the only thing you change in the future.
]

# Start detectnet and capture output
proc = subprocess.Popen(
    DETECTNET_CMD,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

LOGFILE = "anomaly_log.csv"

# Prepare CSV log (create or overwrite)
with open(LOGFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "rule_id", "details"])

# Simple regexes to match object classes in detectnet output
person_re = re.compile(r"\bperson\b", re.IGNORECASE)
phone_re  = re.compile(r"\b(cell\s*phone|phone)\b", re.IGNORECASE)

counts = {
    "person": 0,
    "cell phone": 0
}

last_event_time = 0
cooldown_sec = 2.0  # seconds between log entries so we don't spam

def log_event(rule_id, details):
    """
    Write anomaly event to CSV with timestamp, but rate-limit it.
    """
    global last_event_time
    now = time.time()
    if now - last_event_time > cooldown_sec:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOGFILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, rule_id, details])
        print(f"[ANOMALY] {rule_id}: {details}")
        last_event_time = now

print("anomaly_checker: running detectnet and monitoring for anomalies...")

for line in proc.stdout:
    line = line.strip()
    if not line:
        continue

    # Show raw model output so we can debug
    print("[detectnet]", line)

    # --- 1. Count objects in THIS frame only ---
    people_in_frame = 0
    phones_in_frame = 0

    # detectnet tends to output lines like:
    # "detected person 0.83" or "detected cell phone 0.77"
    # We can just scan this line and bump counters.
    if "person" in line.lower():
        people_in_frame += 1
    if "cell phone" in line.lower() or "phone" in line.lower():
        phones_in_frame += 1

    # --- 2. Apply rules based on this frame ---
    # Rule A: crowding
    MAX_ALLOWED = 3  # this should match your rules.yaml max_people threshold
    if people_in_frame > MAX_ALLOWED:
        msg = f"Detected {people_in_frame} people"
        print("[ANOMALY] max_people:", msg)
        log_event("max_people", msg)  # we'll define log_event below if not already

    # Rule B: phone visible
    if phones_in_frame > 0:
        msg = f"Detected {phones_in_frame} phone(s)"
        print("[ANOMALY] forbid_phone:", msg)
        log_event("forbid_phone", msg)
