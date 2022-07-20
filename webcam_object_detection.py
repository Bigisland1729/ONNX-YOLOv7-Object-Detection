import cv2
import time
import argparse

from YOLOv7 import YOLOv7

def clip(size, x1, y1, x2, y2):
    height, width = size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(height, x2)
    y2 = min(width, y2)
    return x1, y1, x2, y2

def mosaic(src, ratio, boxes):
    dst = src.copy()
    for box in boxes:
        x1, y1, x2, y2 = clip(src.shape[:2][::-1], *(box.astype(int)))
        # print(x1, "", y1, "", x2, "", y2) # debug
        dst[y1:y2, x1:x2] = mosaic1(dst[y1:y2, x1:x2], ratio)
    return dst

def mosaic1(src, ratio):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "models/yolov7-tiny_256x320.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

parser = argparse.ArgumentParser()
parser.add_argument('--mosaic', '-m', type=float, default=None)
opt = parser.parse_args()

if opt.mosaic:
    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.time()

        # Update object localizer
        boxes, scores, class_ids = yolov7_detector(frame)

        elapsed_time = time.time() - start_time

        combined_img = mosaic(frame, opt.mosaic, boxes)

        text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000) + 'ms'
        combined_img = cv2.putText(
            combined_img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            thickness=2,
        )

        cv2.imshow("Detected Objects", combined_img)

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.time()

        # Update object localizer
        boxes, scores, class_ids = yolov7_detector(frame)

        elapsed_time = time.time() - start_time

        combined_img = yolov7_detector.draw_detections(frame)

        text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000) + 'ms'
        combined_img = cv2.putText(
            combined_img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            thickness=2,
        )

        cv2.imshow("Detected Objects", combined_img)

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
