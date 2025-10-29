# 🧠 Real-Time Anomaly Detection on Jetson Nano
**Course:** DATA 690 – Special Topics in AI  
**Student:** Soumya Bhate
**Instructor:** Prof. Levan Sulimanov
---

## 📍 Project Overview
Normal Jetson demos only draw bounding boxes.  
This project extends that into a **real-time safety assistant** that detects anomalies and logs them for review.

### 🎯 Objectives
- Use Jetson Nano’s camera for live object detection  
- Detect people / objects using a pretrained model (`ssd-mobilenet-v2`)  
- Apply simple safety rules (crowding, phone usage, etc.)  
- Log anomalies with timestamps to CSV  

---

## ⚙️ System Architecture
**Hardware**
- NVIDIA Jetson Nano  
- Logitech USB Camera  
- JetPack + Jetson-Inference repo (built locally)

**Software Flow**
1. **`detectnet`** from NVIDIA’s Jetson-Inference performs object detection on live frames.  
2. **`anomaly_checker.py`** (our script) launches `detectnet` as a subprocess and parses its console output.  
3. It checks each frame for rule violations:  
   - **max_people:** > 3 people → crowding alert  
   - **forbid_phone:** phone visible → policy alert  
4. Violations are printed and logged to `anomaly_log.csv`.

---

## 🧩 Project Structure
```
jetson-inference/
 ├─ build/aarch64/bin/                # Compiled Jetson-Inference binaries
 ├─ python/examples/
 │   ├─ detectnet.py
 │   ├─ anomaly_checker.py            # Our anomaly monitoring script
 │   ├─ rules.yaml                    # Policy definitions
 │   ├─ anomaly_log.csv               # Auto-generated log
 │   ├─ session_capture.mp4           # Optional output video
 │   └─ README.md                     # (this file)
```

---

## 🚀 How to Run

### 1️⃣ Plug in Camera
Connect the USB camera (Logitech C270 or similar).

### 2️⃣ Test Object Detection Manually
```bash
cd ~/jetson-inference/build/aarch64/bin
./detectnet --camera=/dev/video0 --input-width=640 --input-height=480
```
If you see bounding boxes → your Jetson + camera + CUDA stack are ready.  
If not, run `ls /dev/video*` and adjust to the correct device (e.g. `/dev/video1`).

---

### 3️⃣ Launch Container Environment
Run this from the Jetson host terminal:

```bash
sudo docker run -it --rm   --runtime nvidia   --network host   --privileged   -e DISPLAY=$DISPLAY   -v /tmp/argus_socket:/tmp/argus_socket   -v /tmp/.X11-unix:/tmp/.X11-unix   -v /dev/video0:/dev/video0   -v /home/ms77930/jetson-inference:/jetson-inference   -v /home/ms77930/jetson-inference/build/aarch64:/arch-build   -v /home/ms77930/recordings:/recordings   -v /usr/lib/python3.8/dist-packages:/usr/lib/python3.8/dist-packages   -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu   nvcr.io/nvidia/l4t-ml:r35.2.1-py3   /bin/bash
```

After the prompt changes to `root@soumya:/#` (or similar according to your username), you’re inside the container.

---

### 4️⃣ Verify DetectNet Runtime
```bash
export LD_LIBRARY_PATH=/arch-build/bin:/arch-build/lib:$LD_LIBRARY_PATH
python3 /arch-build/bin/detectnet.py /dev/video0
```
If it opens a video window with detections, everything is working.

---

### 5️⃣ Run the Anomaly Checker
```bash
cd /jetson-inference/python/examples
export LD_LIBRARY_PATH=/arch-build/bin:/arch-build/lib:$LD_LIBRARY_PATH
python3 anomaly_checker.py
```

This script starts `detectnet`, parses its output, and writes alerts to `anomaly_log.csv`.

---

## 📊 Sample Console Output
```text
[detectnet] detected person 0.83
[detectnet] detected cell phone 0.72
[ANOMALY] max_people: Detected 5 people
[ANOMALY] forbid_phone: Detected 1 phone(s)
```

---

## 📑 CSV Log Example
| timestamp | rule_id | details |
|------------|----------|----------|
| 2025-10-28 14:22:03 | max_people | Detected 5 people |
| 2025-10-28 14:22:06 | forbid_phone | Detected 1 phone(s) |

---

## 🧰 Troubleshooting

| Issue | Cause | Fix |
|-------|--------|-----|
| `videoSource -- failed to create input stream` | Camera index wrong | Run `ls /dev/video*` → use the existing device (e.g. /dev/video1) |
| `cannot open shared object file` | Missing library path | Run `export LD_LIBRARY_PATH=/arch-build/bin:/arch-build/lib:$LD_LIBRARY_PATH` |
| `failed to find model manifest (models.json)` | Model not auto-found | Use explicit model flags:<br>`--model=/jetson-inference/build/aarch64/bin/networks/SSD-Mobilenet-v2/ssd-mobilenet-v2.onnx` and `--labels=.../labels.txt` |
| `Permission denied` | Container not privileged | Use `--privileged` in docker command |

---

## 🧭 Future Work
- Add audio / LED alerts for critical events  
- Integrate a Flask dashboard for live visualization  
- Extend `rules.yaml` for dynamic policy sets  
- Record cropped anomaly snapshots via `jetson_utils.videoOutput()`  

---

## 🙏 Acknowledgements
- Developed as part of the mini project for the course **DATA 690 – Special Topics in AI (UMBC)**.
- Under the guidance of **Prof. Levan Sulimanov**.  
- Built on **NVIDIA Jetson Nano**, **JetPack 5.x**, and the **Jetson-Inference** framework.  
