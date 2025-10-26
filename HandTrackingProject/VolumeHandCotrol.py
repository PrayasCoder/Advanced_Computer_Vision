import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Camera setup
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Hand Detector
detector = htm.handDetector(detectionCon=0.7)

# Audio control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
isMuted = False
lastVol = -20.0  # Default safe volume level
muteStartTime = None  # Timer for mute gesture hold

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Thumb and index finger
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw volume control visuals
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Volume interpolation
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        # Mute/Unmute gesture detection (Index + Middle)
        x3, y3 = lmList[12][1], lmList[12][2]
        middle_dist = math.hypot(x2 - x3, y2 - y3)

        if middle_dist < 40:  # Fingers are close
            if muteStartTime is None:
                muteStartTime = time.time()
            elif time.time() - muteStartTime > 1:  # Held for 1 sec
                isMuted = not isMuted
                if isMuted:
                    volume.SetMute(1, None)
                else:
                    volume.SetMute(0, None)
                muteStartTime = None
        else:
            muteStartTime = None

        # Only set volume if not muted
        if not isMuted:
            volume.SetMasterVolumeLevel(vol, None)
            lastVol = vol  # Remember last safe volume
        else:
            volume.SetMasterVolumeLevel(lastVol, None)

        # Draw green circle if fingers close
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    else:
        # Safe Volume Mode: hold last volume if hand leaves frame
        if not isMuted and lastVol != -20.0:
            volume.SetMasterVolumeLevel(lastVol, None)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Show mute status
    if isMuted:
        cv2.putText(img, "MUTED", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
