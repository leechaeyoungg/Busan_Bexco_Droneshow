import os
from ultralytics import YOLO
import cv2
import numpy as np

# 모델 경로 (포즈 추정 모델 사용)
model_path = r"D:\yolo11l-pose.pt"  
input_video_path = r"D:\detect_person_video\3206888-uhd_3840_2160_30fps.mp4"
output_video_path = r"D:\detect_person_video\3206888-uhd_3840_2160_30fps_pose.mp4"

# COCO 포맷에 따른 키포인트 연결 정보 및 색상
skeleton = {
    "face": [(0, 1), (1, 3), (0, 2), (2, 4)],
    "arms": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)],
    "legs": [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
    "torso": [(5, 11), (6, 12), (5, 6), (11, 12)]
}

colors = {
    "face": (0, 255, 0),  # 초록
    "arms": (255, 0, 0),  # 파랑
    "legs": (0, 0, 255),  # 빨강
    "torso": (0, 255, 255)  # 노랑
}

# 모델 로드
model = YOLO(model_path)

# 객체 트래킹을 위한 딕셔너리
paths = {}

# 영상 로드
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Info - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {frame_count}")

# 결과 저장용 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0  # 현재 프레임 인덱스

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_idx += 1
    print(f"Processing frame {frame_idx}/{frame_count}")

    # 객체 탐지 및 트래킹 수행
    track_results = model.track(frame, persist=True, conf=0.5, iou=0.5, save=False, verbose=False)

    # 포즈 추정 수행
    pose_results = model.predict(frame, conf=0.5, save=False, verbose=False)

    # 현재 프레임 내 객체 수
    object_count = 0

    if track_results and hasattr(track_results[0], 'boxes'):
        for box in track_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_id = int(box.id[0]) if box.id is not None else None

            # 바운딩 박스 및 ID 표시
            label = f"ID {object_id}" if object_id is not None else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 객체 이동 경로 저장 및 트래킹 선 그리기
            if object_id is not None:
                if object_id not in paths:
                    paths[object_id] = []
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                paths[object_id].append(center)

                # 이동 경로 그리기 (이전 위치와 현재 위치 연결)
                for i in range(1, len(paths[object_id])):
                    if paths[object_id][i - 1] is None or paths[object_id][i] is None:
                        continue
                    cv2.line(frame, paths[object_id][i - 1], paths[object_id][i], (255, 255, 0), 3)

            object_count += 1

    # 포즈 키포인트 표시
    if pose_results and hasattr(pose_results[0], 'keypoints'):
        for result in pose_results:
            keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else np.array(result.keypoints.xy)

            for obj_keypoints in keypoints:
                for x, y in obj_keypoints:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)

                # 키포인트 연결
                for part, connections in skeleton.items():
                    color = colors[part]
                    for start, end in connections:
                        if start < len(obj_keypoints) and end < len(obj_keypoints):
                            x1, y1 = obj_keypoints[start]
                            x2, y2 = obj_keypoints[end]
                            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # 객체 수를 화면에 표시 (텍스트 크기 키움)
    text = f"Count: {object_count}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = 30
    text_y = 80
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (255, 255, 255), -1)
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # 결과 프레임 저장
    out.write(frame)

# 리소스 정리
cap.release()
out.release()

print(f"Processed video saved at: {output_video_path}")


