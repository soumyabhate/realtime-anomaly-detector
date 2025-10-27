#!/usr/bin/env python3
#
# detector_monitor.py
#
# Direct detectNet inference + anomaly logging
# (no stdout parsing, we inspect detections directly)

import time
import csv
import os

import jetson.inference   # comes from jetson-inference build
import jetson.utils       # camera/video IO utils

# ----------------- config -----------------

CAMERA_SOURCE = "/dev/video0"    # camera device
ANOMALY_LOG   = "anomaly_log.csv"
MAX_ALLOWED_PEOPLE = 3           # >3 triggers max_people rule
COOLDOWN_SEC = 2.0               # rate limit anomaly spam

# which class names count as "person" or "phone"
PERSON_LABELS = ["person", "people", "man", "woman", "person(s)"]
PHONE_LABELS  = ["cell phone", "phone", "mobile phone", "cellphone"]

# pick a model - this matches the default SSD-Mobilenet-V2 coco model
NET_MODEL = "ssd-mobilenet-v2"
OVERLAY   = "box,labels,conf"     # draw boxes + labels
THRESHOLD = 0.5                   # min confidence

last_event_time = 0.0

# ----------------- helpers -----------------

def log_event(rule_id, details):
    """append one anomaly row (timestamp, rule, details) to CSV,
       but rate-limit so we don't spam every frame."""
    global last_event_time
    now = time.time()
    if now - last_event_time < COOLDOWN_SEC:
        return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(ANOMALY_LOG)

    with open(ANOMALY_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "rule_id", "details"])
        w.writerow([ts, rule_id, details])

    print(f"[ANOMALY] {rule_id}: {details}")
    last_event_time = now


# ----------------- main -----------------

def main():
    print("[monitor] starting detectNet in Python mode...")

    # init network
    net = jetson.inference.detectNet(NET_MODEL, threshold=THRESHOLD)
    # init camera
    cam = jetson.utils.videoSource(CAMERA_SOURCE)      # /dev/video0
    # init display (for live preview on Nano screen)
    disp = jetson.utils.videoOutput("display://0")     # X screen 0

    print("[monitor] running...  (Ctrl+C to stop)")
    while disp.IsStreaming() and cam.IsStreaming():
        # grab frame from camera
        img = cam.Capture()

        # run detection on frame
        detections = net.Detect(img, overlay=OVERLAY)

        # per-frame counters
        people_in_frame = 0
        phones_in_frame = 0

        # inspect every detection box
        for det in detections:
            class_id    = det.ClassID
            class_label = net.GetClassDesc(class_id).lower()  # e.g. "person", "cell phone"

            # debug per box
            print(f"[detect] {class_label} conf={det.Confidence:.2f}")

            # tally people
            for p in PERSON_LABELS:
                if p in class_label:
                    people_in_frame += 1
                    break

            # tally phones
            for ph in PHONE_LABELS:
                if ph in class_label:
                    phones_in_frame += 1
                    break

        # show debug summary for this frame
        print(f"[frame] people={people_in_frame}  phones={phones_in_frame}")

        # ---- apply rules ----

        # Rule: crowd too big
        if people_in_frame > MAX_ALLOWED_PEOPLE:
            msg = f"Detected {people_in_frame} people"
            log_event("max_people", msg)

        # Rule: phone visible
        if phones_in_frame > 0:
            msg = f"Detected {phones_in_frame} phone(s)"
            log_event("forbid_phone", msg)

        # render preview frame to display
        disp.Render(img)
        disp.SetStatus(f"people={people_in_frame} phone={phones_in_frame}")

    print("[monitor] stopped.")


if __name__ == "__main__":
    main()
