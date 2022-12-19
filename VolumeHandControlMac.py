import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import applescript

minVol = 0
maxVol = 100
vol = applescript.run('get output volume of (get volume settings)').out
volBar = np.interp(vol, [0, 100], [400, 150])
volPer = np.interp(vol, [0, 100], [0, 100])

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(detectionCon=0.7)


def SetVolumeMac(vol: int):
    if vol > 100 or vol < 0:
        raise ValueError("vol should be in range 0~100")
    applescript.run(f'set volume output volume {vol}')


while True:
    success, img = cap.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[4], lmList[8]) # 4 for thumb tip and 8 for index finger tip

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2-x1, y2-y1)  # sqrt( (x2-x1)^2 + (y2-y1)^2 )
            # print(length)
            if length < 30:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            # Hand Range 30~150
            # Volume Range 0~100
            vol = np.interp(length, [30, 150], [minVol, maxVol])
            volBar = np.interp(length, [30, 150], [400, 150])
            volPer = np.interp(length, [30, 150], [0, 100])
            # print(vol)
            SetVolumeMac(vol)
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400),
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"volume%:{int(volPer)}", (40, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("img", img)

    if cv2.waitKey(1) == ord('q'):
        break
