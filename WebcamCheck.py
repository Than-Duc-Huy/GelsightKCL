import cv2
import time
import numpy as np

RANGE = 10
all_frame = None
for i in range(RANGE):
    cam = cv2.VideoCapture(i)
    if cam.isOpened():
        start = time.time()
        while time.time() - start < 1:
            frame = cam.read()[1]
        cv2.putText(
            frame, f"Index: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        frame = cv2.resize(frame, (400, 400))
        all_frame = np.hstack((all_frame, frame)) if all_frame is not None else frame
cv2.imshow("All", all_frame)
cv2.waitKey(500)
index = int(input("Type in camera index: "))
cv2.destroyAllWindows()

cam = cv2.VideoCapture(index)
while True:
    frame = cam.read()[1]
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
