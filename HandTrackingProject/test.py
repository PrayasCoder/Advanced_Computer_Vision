import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ------------------ Setup Pycaw ------------------
# Get the default audio device
device = AudioUtilities.GetSpeakers()
volume_interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(volume_interface, POINTER(IAudioEndpointVolume))

# Get min and max volume levels
vol_min, vol_max, _ = volume.GetVolumeRange()

# ------------------ Setup Mediapipe ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Thumb tip and index finger tip coordinates
        h, w, _ = frame.shape
        thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        # Draw circles
        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Calculate distance between thumb and index
        distance = get_distance((x1, y1), (x2, y2))

        # Map distance to volume
        vol_range = vol_max - vol_min
        vol_level = vol_min + (distance/200) * vol_range  # 200 is max pixel distance
        vol_level = max(min(vol_level, vol_max), vol_min)
        volume.SetMasterVolumeLevel(vol_level, None)

        # Show volume bar
        cv2.rectangle(frame, (50, 400), (85, 150), (0, 255, 0), 2)
        vol_bar = int(((vol_level - vol_min) / vol_range) * 250)
        cv2.rectangle(frame, (50, 400-vol_bar), (85, 400), (0, 255, 0), -1)

    cv2.imshow("Volume Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
