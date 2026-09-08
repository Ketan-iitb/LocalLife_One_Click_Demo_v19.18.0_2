#!/usr/bin/env python3
"""
Windows Depth Anything V2 worker for LocalLife V14.

The worker:
1. fetches the latest Logitech C920 image from the Pi,
2. runs Depth Anything V2,
3. keeps the V14 dashboard informed that the DA worker is online.

This recreation deliberately does NOT fabricate a liters value from monocular
depth without an explicit metric calibration. Add a calibrated RGB-volume
mapping later if/when you have reference measurements.
"""

import argparse
import time
import requests
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

parser = argparse.ArgumentParser()
parser.add_argument("--pi", default="http://192.168.0.123:5005")
args = parser.parse_args()

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

print("Loading:", MODEL_ID)
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

print("Depth Anything V2 worker online on:", device)
print("Pi:", args.pi)

while True:
    try:
        r = requests.get(args.pi + "/api/da_input", timeout=5)
        if r.status_code != 200:
            time.sleep(1)
            continue

        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs).predicted_depth

        depth = torch.nn.functional.interpolate(
            output.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()

        finite = depth[np.isfinite(depth)]
        med = float(np.median(finite)) if finite.size else float("nan")

        # In recreated V14 we only report worker-online status unless metric
        # RGB volume has been calibrated. This prevents false liters values.
        requests.post(
            args.pi + "/api/da_result",
            json={"volume_l": None},
            timeout=5,
        )

        print("DA frame OK | median depth:", round(med, 3))
        time.sleep(0.10)

    except KeyboardInterrupt:
        print("Stopping V14 DA worker.")
        break
    except Exception as exc:
        print("Retry:", exc)
        time.sleep(2)
