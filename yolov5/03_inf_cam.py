#!/usr/bin/env python
# D021 YOLOv4 Camera


import torch
import cv2

model = torch.hub.load("Ultralytics/yolov5", "yolov5m")
print(f"Classes: {model.names}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(rgb_frame)

    for i, obj in enumerate(results.xyxy[0]):
        x1, y1, x2, y2, _, cls = map(int, obj[:6])
        conf = obj[4]
        label_str = f"{model.names[cls]} : {conf:.2f}"
        # print(f"SKY: {x1}, {y1}, {x2}, {y2}, {conf}, {model.names[cls]}")

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label_str, (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("detected", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
