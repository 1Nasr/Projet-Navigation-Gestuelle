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

# Parametres pointeur pour fluidite
LISSAGE_POINTEUR = 0.35      # 0..1, plus grand = plus reactif
LISSAGE_POINTEUR_RAPIDE = 0.6
SEUIL_ACCELERATION_PX = 40
ZONE_MORTE_POINTEUR_PX = 2       # ignore micro mouvements pour eviter le jitter
INTERVALLE_MAJ_POINTEUR = 0.005  # 200 Hz max
NB_FRAMES_INDEX_STABLE = 1
NB_FRAMES_POING_MODE = 4
COOLDOWN_CHANGEMENT_MODE = 0.8

derniere_position_pointeur = None
dernier_maj_pointeur = 0.0
serie_index_leve = 0
mode_actif = "swipe"
compteur_poing_ferme = 0
dernier_changement_mode = 0.0

pyautogui.PAUSE = 0

def is_index_up(hand):
    index_up  = hand[8].y < hand[6].y   # index levé
    middle_down = hand[12].y > hand[10].y  # majeur fermé
    ring_down   = hand[16].y > hand[14].y  # annulaire fermé
    pinky_down  = hand[20].y > hand[18].y  # auriculaire fermé
    return index_up and middle_down and ring_down and pinky_down

def is_fist_closed(hand):
    index_down = hand[8].y > hand[6].y
    middle_down = hand[12].y > hand[10].y
    ring_down = hand[16].y > hand[14].y
    pinky_down = hand[20].y > hand[18].y
    return index_down and middle_down and ring_down and pinky_down
    
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

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        swipe_text = ""

        if last_result and last_result.hand_landmarks:
            hand = last_result.hand_landmarks[0]
            index_en_haut = is_index_up(hand)
            poing_ferme = is_fist_closed(hand)
            poignet = hand[0]

            # Changement de mode avec poing ferme
            if poing_ferme:
                compteur_poing_ferme += 1
            else:
                compteur_poing_ferme = 0

            if (
                compteur_poing_ferme >= NB_FRAMES_POING_MODE
                and (now - dernier_changement_mode) >= COOLDOWN_CHANGEMENT_MODE
            ):
                mode_actif = "pointeur" if mode_actif == "swipe" else "swipe"
                dernier_changement_mode = now
                compteur_poing_ferme = 0
                positions.clear()
                serie_index_leve = 0

            # Dessin des lignes et points sur l'image 
            for lm in hand:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,0), -1)
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, (int(hand[s].x*w), int(hand[s].y*h)), (int(hand[e].x*w), int(hand[e].y*h)), (255,0,0), 2)

            # Mode pointeur et mode swipe sont mutuellement exclusifs
            if mode_actif == "pointeur":
                positions.clear()
                if index_en_haut:
                    serie_index_leve += 1
                else:
                    serie_index_leve = 0

                if serie_index_leve >= NB_FRAMES_INDEX_STABLE and (now - dernier_maj_pointeur) >= INTERVALLE_MAJ_POINTEUR:
                    # Position du bout de l'index (landmark 8)
                    bout_index = hand[8]

                    # Mapper les coords webcam (0.0→1.0) vers l'ecran
                    cible_x = int((1 - bout_index.x) * SCREEN_W)  # miroir horizontal
                    cible_y = int(bout_index.y * SCREEN_H)

                    if derniere_position_pointeur is None:
                        x_lisse, y_lisse = cible_x, cible_y
                    else:
                        x_precedent, y_precedent = derniere_position_pointeur
                        delta_x = abs(cible_x - x_precedent)
                        delta_y = abs(cible_y - y_precedent)
                        coeff_lissage = LISSAGE_POINTEUR_RAPIDE if (delta_x + delta_y) > SEUIL_ACCELERATION_PX else LISSAGE_POINTEUR
                        x_lisse = int(x_precedent + (cible_x - x_precedent) * coeff_lissage)
                        y_lisse = int(y_precedent + (cible_y - y_precedent) * coeff_lissage)

                    # Evite d'envoyer des moves trop petits et couteux
                    if (
                        derniere_position_pointeur is None
                        or abs(x_lisse - derniere_position_pointeur[0]) >= ZONE_MORTE_POINTEUR_PX
                        or abs(y_lisse - derniere_position_pointeur[1]) >= ZONE_MORTE_POINTEUR_PX
                    ):
                        pyautogui.moveTo(x_lisse, y_lisse)
                        derniere_position_pointeur = (x_lisse, y_lisse)
                        dernier_maj_pointeur = now

                    swipe_text = "POINTEUR ACTIF"
            else:
                serie_index_leve = 0

                # Position poignet
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

        else:
            positions.clear()
            serie_index_leve = 0
            compteur_poing_ferme = 0

        # Affichage texte lors de la détection pour le débug (a terme on l'enlevera)
        if swipe_text:
            cv2.putText(frame, swipe_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)

        cv2.putText(frame, f"MODE: {mode_actif.upper()}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("PowerPoint Swipe with your hand :)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
