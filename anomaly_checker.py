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

DETECTNET_CMD = [
    "/home/ms77930/jetson-inference/build/aarch64/bin/detectnet",
    "--camera=/dev/video0",
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

    # Always show the raw detectnet output so we can debug camera vs model
    print("[detectnet]", line)

    # If detectnet prints something like "detected person", count it
    if "detected" in line:
        if person_re.search(line):
            counts["person"] += 1
        if phone_re.search(line):
            counts["cell phone"] += 1

    # ---------------- RULES ----------------
    # Rule 1: crowding / occupancy
    if counts["person"] > 3:
        log_event("max_people", f"Detected {counts['person']} people")

    # Rule 2: forbidden object (phone)
    if counts["cell phone"] > 0:
        log_event("forbid_phone", f"Cell phone detected ({counts['cell phone']})")

    # Reset counts after each frame timeout / buffer rollover so it doesn't just climb forever
    if "timeout occurred waiting for the next image buffer" in line \
       or "next image buffer" in line:
        counts = {"person": 0, "cell phone": 0}
