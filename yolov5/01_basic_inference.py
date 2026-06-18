import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Images
img = "https://ultralytics.com/images/zidane.jpg"

# Inference
results = model(img)

# Results
results.show()
