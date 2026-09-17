import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Finger tips
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]
            ring = landmarks[16]
            pinky = landmarks[20]

            # Finger joints
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            ring_pip = landmarks[14]
            pinky_pip = landmarks[18]

            fingers = [
                index.y < index_pip.y,
                middle.y < middle_pip.y,
                ring.y < ring_pip.y,
                pinky.y < pinky_pip.y
            ]

            if all(fingers):
                gesture = "OPEN PALM"

            elif not any(fingers):
                gesture = "FIST"

            elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                gesture = "PEACE"

            elif thumb.y < index_pip.y and not any(fingers):
                gesture = "THUMBS UP"

    cv2.putText(
        frame,
        gesture,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("AI Gesture Recognition", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()