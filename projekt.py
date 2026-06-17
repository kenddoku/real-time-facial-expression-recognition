import cv2
import mediapipe as mp
import os
from datetime import datetime

SAVE_DIR = "faces"
os.makedirs(SAVE_DIR, exist_ok=True)

face = mp.solutions.face_detection
face_detection = face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("camera does not work")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #prepearing for mediapipe
    frame = cv2.flip(frame, 1) #lustrzane odbicie, zeby naturalnie było
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #for opencv and mediapipe

    results = face_detection.process(rgb)
    face_crop = None

    if results.detections:
        #we have pixel matrix
        h, w, channels = frame.shape #height width,  channels (of colors)

        for detection in results.detections:
            box = detection.location_data.relative_bounding_box 

            #coordinates in pixels
            x = int(box.xmin * w)  #box.min and box.max cooridnates in %
            y = int(box.ymin * h)
            #
            bw = int(box.width * w)
            bh = int(box.height * h)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            face_crop = frame[y:y+bh, x:x+bw]

    cv2.imshow("kamera", frame)

    key = cv2.waitKey(1) & 0xFF

    #quit program
    if key == ord('q'):
        break
    # savign face
    if key == ord('s') and face_crop is not None and face_crop.size > 0:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, face_crop)
        print(f"saved!!!!: {path}")

cap.release()
cv2.destroyAllWindows()