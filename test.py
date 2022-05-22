import cv2
from cv2 import CAP_DSHOW

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# webcam.set(3, 640)
# webcam.set(4, 480)
while True:
    suc, img = webcam.read()
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()