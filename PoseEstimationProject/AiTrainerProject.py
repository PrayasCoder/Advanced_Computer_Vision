import cv2
import numpy as np
import time
import PoseModule as pm


cap = cv2.VideoCapture("AiTrainer/curls.mp4")
detector = pm.poseDetector

while True:
    # success, img = cap.read()
    # img = cv2.resize(img, (1280, 720))
    img = cv2.imread("AiTrainer/test.jpg")

    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)
    # print(lmList)

    if len(lmList) != 0 :
                

    cv2.imshow("Image", img)
    cv2.waitKey(1)
