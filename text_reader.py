import cv2
import easyocr

reader = easyocr.Reader(['en'])

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "Press S to Scan Text | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Text Reader", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("Scanning text...")

        cv2.imwrite("scan.jpg", frame)

        results = reader.readtext("scan.jpg")

        if results:
            for box, text, confidence in results:
                print("TEXT:", text)
        else:
            print("No text detected")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()