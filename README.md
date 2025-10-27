# Real-Time Anomaly Detection on Jetson Nano
*Subject:* DATA 690 Special Topics in AI  
*Student:* Soumya Bhate  
*Campus ID:* MS77930

## 1. Problem
Normal Jetson demos just draw bounding boxes.  
My goal was to turn that into something useful in the real world: a live safety/monitoring assistant.

The system should:
- Watch a camera feed in real time.
- Detect people / objects using a pretrained model (no retraining).
- Check custom safety rules (like "too many people" or "phone detected").
- Log violations with timestamps for review.

This can be used for lab safety, basic compliance, or occupancy monitoring.

## 2. System Overview
Hardware:
- NVIDIA Jetson Nano
- Logitech USB camera
- JetPack + jetson-inference repo built locally

Software pipeline:
1. detectnet (from NVIDIA's jetson-inference) runs real-time object detection on the camera.
2. anomaly_checker.py (my code) launches detectnet as a subprocess and reads its live output.
3. For each frame, my code:
   - Counts people
   - Looks for "cell phone"
   - Applies rules:
     - max_people: more than 3 people → crowding alert
     - forbid_phone: phone visible → policy alert
4. When a rule triggers:
   - An [ANOMALY] ... alert is printed on the console
   - A row is appended to anomaly_log.csv with timestamp, rule id, and details

So instead of just "here are boxes", it becomes "here is a log of unsafe events."

## 3. How to Run (Jetson Nano)
Step 1. Plug in camera to the Jetson Nano.

Step 2. Test the object detector manually:
```bash
cd ~/jetson-inference/build/aarch64/bin
./detectnet --camera=/dev/video0 --input-width=640 --input-height=480

