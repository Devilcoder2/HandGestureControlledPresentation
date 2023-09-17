import cv2
import os 
from cvzone.HandTrackingModule import HandDetector
import numpy as np

#Variables 
width, height = 1280, 1080
folderPath = "Presentation"

#Camera setup
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#Importing images
pathImages = sorted(os.listdir(folderPath), key=len)

imageNumber = 0
hs, ws = 240, 416
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0 
buttonDelay = 10
annotations = [[]]
annotationNumber = 0
annotationStart = False

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1) 
    pathFullImg = os.path.join(folderPath, pathImages[imageNumber])
    imgCurrent = cv2.imread(pathFullImg)
    
    hands, img = detector.findHands(img)
    imgCurrentSmall = cv2.resize(imgCurrent,(1280,720))
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0,255,0), 10)
    
    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx,cy = hand['center']
        lmList = hand['lmList']
        
        #Constrain values for easy drawing
        xVal = int(np.interp(lmList[8][0], [ws//2, ws], [0,width]))
        yVal = int(np.interp(lmList[8][1], [hs//2 + 100, hs+100], [0,height]))
        indexFinger = xVal, yVal
    
        if cy<=gestureThreshold: #if hand is at the height of the face
            annotationStart = False
            # Gesture 1- Left 
            if fingers == [1,0,0,0,0]:
                annotationStart = False
                if imageNumber > 0:
                    imageNumber -= 1
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                   
                
            # Gesture 2 - Right
            if fingers == [0,0,0,0,1]:  
                annotationStart = False
                if imageNumber < len(pathImages)-1:
                    imageNumber += 1
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    
            
        # Gesture 3 - Show Pointer
        if fingers == [0,1,1,0,0]:
            cv2.circle(imgCurrentSmall, indexFinger, 12, (0,0,255), cv2.FILLED)
            annotationStart = False
            
         # Gesture 4 - Draw Pointer
        if fingers == [0,1,0,0,0]:
            if annotationStart is False:
                annotationStart = True 
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrentSmall, indexFinger, 12, (0,0,255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False
    
        # Gesture 5 - Erase
        if fingers == [0,1,1,1,0]:
            if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    else: 
        annotationStart = False
         
    #Button Pressed iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False
    
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j!=0:
                cv2.line(imgCurrentSmall, annotations[i][j-1], annotations[i][j], (0,0,200), 12)
    
    #Adding webcam image on the slides
    imgSmall = cv2.resize(img,(ws,hs))
    h, w, _ = imgCurrentSmall.shape
    imgCurrentSmall[0:hs,w-ws:w] = imgSmall
    
    cv2.imshow("Image", img)
    cv2.imshow("Presentation", imgCurrentSmall)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()