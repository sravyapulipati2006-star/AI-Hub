import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    danger = False

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]

        if name in ["knife", "fire", "car"] and confidence > 0.5:
            danger = True

    if danger:
        cv2.putText(
            annotated,
            "!!! SAFETY ALERT !!!",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    else:
        cv2.putText(
            annotated,
            "SAFE",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    cv2.imshow("AI Safety Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()