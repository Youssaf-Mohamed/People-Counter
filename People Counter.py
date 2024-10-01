import cv2
import cvzone 
from ultralytics import YOLO
import numpy as np
import math
from sort import *

tracker = Sort(max_age=20)

def mouseEvent(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Left mouse button clicked at ({x},{y})')

# Define tracking areas based on user input

################################################################
# For example, using mouse click coordinates:
# Left mouse button clicked at (706,384) -> areaLeft[0]
# Left mouse button clicked at (550,444) -> areaLeft[1]
# Left mouse button clicked at (550,444) -> areaRight[0]
# Left mouse button clicked at (279,546) -> areaRight[1]
################################################################


areaLeft = [(706, 384), (550, 444)]  #defined area for left tracking
areaRight = [(550, 444), (279, 546)]  #defined area for right tracking
countRight = set()
countLeft = set()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouseEvent)

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(r"C:\Users\DELL\Downloads\Telegram Desktop\pythone\openCv\vidoies\people.mp4")
mask = cv2.imread(r"C:\Users\DELL\Downloads\Telegram Desktop\pythone\yoloV8\cvzone\mask people counter (5).png")

height = 720
width = 1280

while True:
    ret, frame = cap.read()
    mask = cv2.resize(mask, (width, height))
    frameGen = cv2.bitwise_and(frame, mask)
    results = model(frameGen)

    cv2.line(frame, areaLeft[0], areaLeft[1], (0, 0, 255), 2)
    cv2.circle(frame, areaLeft[0], 3, (0, 0, 255), -1)
    cv2.circle(frame, areaLeft[1], 5, (0, 0, 255), -1)
    cv2.line(frame, areaRight[0], areaRight[1], (255, 0, 255), 2)
    cv2.circle(frame, areaRight[0], 3, (255, 0, 255), -1)
    cv2.circle(frame, areaRight[1], 3, (255, 0, 255), -1)

    img = cv2.imread(r"C:\Users\DELL\Downloads\Telegram Desktop\pythone\yoloV8\cvzone\image copy.png", cv2.IMREAD_UNCHANGED)

    cvzone.overlayPNG(frame, img, (730, 260))

    points = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy()
        confidence = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = math.ceil(confidence[i] * 100) / 100
            points.append([x1, y1, x2, y2, conf])
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            label = f"{model.names[int(labels[i])]} {conf:0.1f}"

    boxes_id = tracker.update(np.array(points))

    for i, box_id in enumerate(boxes_id):
        x1, y1, x2, y2, id = map(int, box_id)
        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1),
                          l=7, colorR=(255, 0, 0), colorC=(0, 0, 255))
        cvzone.putTextRect(frame, str(id), (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=3)
        
        cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2

        cv2.circle(frame, (cx, cy), 2, (255, 0, 255), -1)

        # Check if the center of the box crosses the defined areas
        if areaLeft[1][0] < cx < areaLeft[0][0] and areaLeft[0][1] - 10 < cy < areaLeft[0][1] + 10:
            countLeft.add(id)
            cv2.line(frame, areaLeft[0], areaLeft[1], (0, 255, 0), 2)

        if areaRight[1][0] < cx < areaRight[0][0] and areaRight[0][1] - 10 < cy < areaRight[0][1] + 10:
            countRight.add(id)
            cv2.line(frame, areaRight[0], areaRight[1], (0, 255, 0), 2)

    r = len(countRight)
    l = len(countLeft)
    cv2.putText(frame, str(len(countRight)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(frame, str(len(countLeft)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
