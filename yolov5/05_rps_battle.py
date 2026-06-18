#!/usr/bin/env python
# D021 YOLOv5 Rock Paper Scissors

import torch
import cv2
import os
import json
import warnings

PAPER = 0
ROCK = 1
SCISSORS = 2

# Colors in BGR order.
COLOR_WINNER = (0, 255, 0)
COLOR_LOSER = (0, 0, 255)
COLOR_DRAW = (0, 255, 255)
COLOR_SINGLE = (128, 128, 128)

def decide_winner(player1, player2):
    if player1 == player2:
        return None
    wins = {(ROCK, SCISSORS), (SCISSORS, PAPER), (PAPER, ROCK)}
    return player1 if (player1, player2) in wins else player2


# TODO: 커스텀 모델 로드
model_path = os.path.abspath("noipex_rps.pt")
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)


# Video capture
cap = cv2.VideoCapture(0)

# Loop for camera frames
while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # break the loop on error
    if not ret:
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # Detection 된 내용
    detections = results.xyxy[0]

    # 누가 이겼나
    winner = None

    # 두 사람이 detect되 었을 때
    if len(detections) == 2:
        cls0 = int(detections[0][5])
        cls1 = int(detections[1][5])
        winner = decide_winner(cls0, cls1)

    for i, obj in enumerate(detections):
        x1, y1, x2, y2, _, cls = map(int, obj)
        conf = obj[4]
        label = f"{results.names[cls]}: {conf:.2f}"

        if len(detections) == 2:
            if winner is None:
                color = COLOR_DRAW
            elif cls == winner:
                color = COLOR_WINNER
            else:
                color = COLOR_LOSER
        else:
            # Single detection.
            color = COLOR_SINGLE

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"Object {i}: {label} at [{x1}, {y1}, {x2}, {y2}]")

    # 화면 표시
    cv2.imshow("RPS-Battle", frame)

    # 종료를 위한 key 처리
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
