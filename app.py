#!/usr/bin/env python3
import argparse, time, csv, math, os, sys, yaml
import jetson.inference, jetson.utils

# ---------------- helpers ----------------

def centroid(det):
    # det has .Left, .Top, .Right, .Bottom from jetson.inference
    x = (det.Left + det.Right) / 2.0
    y = (det.Top  + det.Bottom) / 2.0
    return (x, y)

def pairwise_min_distance(points):
    if len(points) < 2:
        return float('inf')
    md = float('inf')
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist = math.hypot(dx, dy)
            if dist < md:
                md = dist
    return md

def load_rules(path):
    with open(path, 'r') as f:
        y = yaml.safe_load(f)
    return y.get('rules', [])

def summarize_detections(detections, net):
    counts = {}
    centroids = {}

    for det in detections:
        cls_name = net.GetClassDesc(det.ClassID)
        counts[cls_name] = counts.get(cls_name, 0) + 1
        centroids.setdefault(cls_name, []).append(centroid(det))

    return counts, centroids

def check_rules(rules, counts, centroids):
    events = []
    for rule in rules:
        rtype = rule['type']
        rid   = rule['id']

        if rtype == 'count_over':
            target_class = rule['class']
            thresh = rule['threshold']
            c = counts.get(target_class, 0)
            if c > thresh:
                events.append((rid, f"{target_class}={c}>{thresh}"))

        elif rtype == 'forbid_class':
            target_class = rule['class']
            c = counts.get(target_class, 0)
            if c > 0:
                events.append((rid, f"forbidden {target_class} seen ({c})"))

        elif rtype == 'require_together':
            req_list = rule['require']  # e.g. ["person", "helmet"]
            have = [counts.get(x, 0) > 0 for x in req_list]
            # if first thing is present but at least one of the others is missing -> event
            if have[0] and (not all(have)):
                events.append((rid, f"missing combo {req_list}"))

        elif rtype == 'min_distance':
            target_class = rule['class']
            limit_px = rule['pixels']
            pts = centroids.get(target_class, [])
            d = pairwise_min_distance(pts)
            if len(pts) >= 2 and d < limit_px:
                events.append((rid, f"{target_class} too close: {d:.1f}px < {limit_px}px"))

    return events

# ---------------- main loop ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--model', default='ssd-mobilenet-v2')  # or your trained model path/name
    parser.add_argument('--overlay', default='boxes,labels,conf')
    parser.add_argument('--rules', default='rules.yaml')
    parser.add_argument('--log',   default='anomaly_log.csv')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # load detector
    net  = jetson.inference.detectNet(args.model, threshold=args.threshold)

    # camera input
    cam  = jetson.utils.videoSource(args.camera)

    # display/output target (X11 window by default, or can be "video.mp4")
    disp = jetson.utils.videoOutput()

    # load rules
    rules = load_rules(args.rules)

    # prepare CSV log file
    if not os.path.exists(args.log):
        with open(args.log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','rule_id','details'])

    last_event_time = 0
    cooldown_sec = 1.0  # seconds to wait before logging the same alert again

    while disp.IsStreaming():
        # 1. Capture a frame
        img = cam.Capture()

        # 2. Run object detection on the frame
        dets = net.Detect(img, overlay=args.overlay)

        # 3. Summarize detections for rule logic
        counts, centroids = summarize_detections(dets, net)

        # 4. Check all rules against what we saw
        events = check_rules(rules, counts, centroids)

        # 5. If rule triggered -> draw alert + log it
        if events:
            # red bar at top
            jetson.utils.cudaDrawRect(img, (0,0,img.width,40), (255,0,0,180))
            # white text on red
            jetson.utils.cudaDrawText(img,
                                      f"ANOMALY: {events[0][0]}",
                                      5, 5,
                                      (255,255,255,255))

            now = time.time()
            if now - last_event_time > cooldown_sec:
                with open(args.log, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for rid, detail in events:
                        writer.writerow([
                            time.strftime('%Y-%m-%d %H:%M:%S'),
                            rid,
                            detail
                        ])
                last_event_time = now

        # 6. Show frame (or write to file if disp was set to "video.mp4" instead)
        disp.Render(img)
        disp.SetStatus(f"Detections: {len(dets)} | Counts: {counts}")

if _name_ == "_main_":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
