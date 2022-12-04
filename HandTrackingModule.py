import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, MaxHands=2, complexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.Maxhands = MaxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.Maxhands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks: # handLms = hand landmarks
                if draw:
                    self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=5)#點的樣式
                    self.handConStyle = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=10)#線的樣式
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.handLmsStyle, self.handConStyle)
        return img       
            
    def findPosition(self, img, handNo=0, draw=True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    imgHeight, imgWidth, imgCenter = img.shape
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    lmList.append([id,xPos,yPos])
                    if draw:
                        cv2.putText(img,str(id),(xPos-25,yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 2)
                        self.mpDraw.draw_landmarks(img, myHand, self.mpHands.HAND_CONNECTIONS, self.handLmsStyle, self.handConStyle)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime=0
    cTime=0
    detector = handDetector()
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

if __name__ == "__main__":
    main()