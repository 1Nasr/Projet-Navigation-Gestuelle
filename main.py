import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        # Dessin des landmarks
        if latest_result and latest_result.hand_landmarks:
            for hand in latest_result.hand_landmarks:

                # Points
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Connexions
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand[start].x * w)
                    y1 = int(hand[start].y * h)
                    x2 = int(hand[end].x * w)
                    y2 = int(hand[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("MediaPipe Hands (Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
