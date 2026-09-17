import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera frame not received")
        break
    print(frame[100,100])

    cv2.imshow("Live Camera", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
    print(len(contours))
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Gray", gray)
    cv2.imshow("Edges", edges)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()