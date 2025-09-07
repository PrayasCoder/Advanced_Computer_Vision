import cv2
import time
import PoseModule as pm

# cap = cv2.VideoCapture('PoseVideos/4.mp4')
cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    # ✅ Resize video frame to fit laptop screen
    img = cv2.resize(img, (600, 1000))

    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if lmList and len(lmList) > 14:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
    # ✅ FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
