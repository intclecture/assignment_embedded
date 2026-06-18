import torch
import cv2

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Capture device
cap = cv2.VideoCapture(0)

# Video frame
ret, bgr_frame = cap.read()
rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

# Inference
results = model(rgb_frame)

# Results
results.show()
cap.release()
cv2.destroyAllWindows()
