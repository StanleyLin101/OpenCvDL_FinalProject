import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime=0
cTime=0
detector = htm.handDetector()
while True:
    success, img = cap.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,f"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv2.imshow("img",img)
    if cv2.waitKey(1) == ord('q'):
            break
