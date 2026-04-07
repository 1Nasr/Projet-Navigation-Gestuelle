import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from collections import deque
import time
import pyautogui #permet "émuler " le clavier et faire les fleche de gauche et de droite 

# ===============================
# MediaPipe setup
# ===============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

#coord des points de la main
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

# ===============================
# Variables
# ===============================

last_result = None
positions = deque(maxlen=8)  # (x, y, time)

MIN_FRAMES = 3
SWIPE_DISTANCE = 0.08     # 8% largeur pour détecter dès petit mouvement
MAX_TIME = 0.4            # secondes
COOLDOWN = 1.2           # secondes

last_swipe_time_left = 0
last_swipe_time_right = 0

SCREEN_W, SCREEN_H = pyautogui.size()

def is_index_up(hand):
    index_up  = hand[8].y < hand[6].y   # index levé
    middle_down = hand[12].y > hand[10].y  # majeur fermé
    ring_down   = hand[16].y > hand[14].y  # annulaire fermé
    pinky_down  = hand[20].y > hand[18].y  # auriculaire fermé
    return index_up and middle_down and ring_down and pinky_down
    
# ===============================
# Callback
# ===============================
def result_callback(result, output_image, timestamp_ms):
    global last_result
    last_result = result

# ===============================
# Options MediaPipe
# ===============================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),  # Fournis sur le site de google fichier qui contient la base d'entrainement pour fonctionner en local
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        now = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        swipe_text = ""

        if last_result and last_result.hand_landmarks:
            hand = last_result.hand_landmarks[0]

            # Dessin des lignes et points sur l'image 
            for lm in hand:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,0), -1)
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, (int(hand[s].x*w), int(hand[s].y*h)), (int(hand[e].x*w), int(hand[e].y*h)), (255,0,0), 2)


            # Position poignet
            poignet = hand[0]
            positions.append((poignet.x, now))

            # Détection swipe dès que dx dépasse seuil
            if len(positions) >= MIN_FRAMES:
                # ici detection de la premiere frame et de la derniere recuper et comparaison de la distance
                x_old, t_old = positions[0]
                x_new, t_new = positions[-1]

                dx = x_new - x_old
                dt = t_new - t_old

                if abs(dx) > SWIPE_DISTANCE and dt < MAX_TIME:
                    if dx > 0 and now - last_swipe_time_left > COOLDOWN:
                        swipe_text = "SWIPE DROITE -> GAUCHE"
                        pyautogui.press('left')
                        print("➡ Swipe détecté : GAUCHE → DROITE")
                        last_swipe_time_left = now
                        positions.clear()
                    elif dx < 0 and now - last_swipe_time_right > COOLDOWN:
                        swipe_text = "SWIPE GAUCHE -> DROITE"
                        pyautogui.press('right')
                        print("⬅ Swipe détecté : DROITE → GAUCHE")
                        last_swipe_time_right = now
                        positions.clear()
                
                if is_index_up(hand):
                    # Position du bout de l'index (landmark 8)
                    index_tip = hand[8]
                    
                    # Mapper les coords webcam (0.0→1.0) vers l'écran
                    screen_x = int((1 - index_tip.x) * SCREEN_W)  # miroir horizontal
                    screen_y = int(index_tip.y * SCREEN_H)
                    
                    pyautogui.moveTo(screen_x, screen_y, duration=0)
                    swipe_text = "POINTEUR ACTIF"

        else:
            positions.clear()

        # Affichage texte lors de la détection pour le débug (a terme on l'enlevera)
        if swipe_text:
            cv2.putText(frame, swipe_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)

        cv2.imshow("PowerPoint Swipe with your hand :)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
