import cv2
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = camera.read()

    if not ret:
        break

    results = model(frame)
    for box in results[0].boxes:
       class_id = int(box.cls[0])
       label = model.names[class_id]
       x1, y1, x2, y2 = map(int, box.xyxy[0])
       print(x1, y1, x2, y2)
       print(model.names[class_id])

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break