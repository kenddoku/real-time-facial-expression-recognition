import cv2
import mediapipe as mp
import os
from datetime import datetime
import numpy as np
import pygame

pygame.mixer.init()

SAVE_DIR = "faces"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_vid(path, bg_color ="black", howbig_pixels = 350, soundon = False, sound_name = "mlg_classic.mp3" ):

    sound = pygame.mixer.Sound("mlg_classic.mp3") 
    if soundon:
        sound.play()
        sound.play(loops=-1)
        #sound.stop()


    """
    bg_color: "black", "white", "blue"
    """
    color_ranges = {
        "black": (np.array([0,   0,   0]),   np.array([180, 255,  40])),
        "green": (np.array([40,  100, 100]), np.array([57,  221, 255])),
        "white": (np.array([0,   0,   180]), np.array([180,  40, 255])),
    }

    cap = cv2.VideoCapture(path)
    frames = []

    lower, upper = color_ranges.get(bg_color, color_ranges["black"]) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # removing black background using masks
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_bg = cv2.inRange(hsv, lower, upper)
        mask_object = cv2.bitwise_not(mask_bg)

        #scaling to num of pixels
        scale = howbig_pixels / frame.shape[1]
        new_w = howbig_pixels
        new_h = int(frame.shape[0] * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        mask_resized = cv2.resize(mask_object, (new_w, new_h))

        frames.append((frame_resized, mask_resized))

    cap.release()
    return frames


def overlay_object(background, cat_frame, cat_mask, margin=10, fullscreen=False):

    bg_h, bg_w = background.shape[:2] #background to cut
    if fullscreen:
        cat_frame = cv2.resize(cat_frame, (bg_w, bg_h))
        cat_mask = cv2.resize(cat_mask, (bg_w, bg_h))
        x_offset, y_offset = 0, 0
    else:
        cat_h, cat_w = cat_frame.shape[:2]
        x_offset = margin
        y_offset = bg_h - cat_h - margin
        if y_offset < 0 or x_offset + cat_w > bg_w:
            return background


    #bg_h, bg_w = background.shape[:2] #background to cut
    cat_h, cat_w = cat_frame.shape[:2]

    bg_new = background[y_offset:y_offset + cat_h, x_offset:x_offset + cat_w]# size of new object window
    alpha = cat_mask.astype(float) / 255.0 #norming
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

    # Blendowanie: kot * maska + tło * (1 - maska)
    blended = (cat_frame.astype(float) * alpha_3ch +
               bg_new.astype(float) * (1.0 - alpha_3ch))

    background[y_offset:y_offset + cat_h, x_offset:x_offset + cat_w] = blended.astype(np.uint8)
    return background

CAT_VIDEO = "cat_vid.mp4"
BUM = "mlg.mp4"
print("Have good time watching the movie! :)")

cat_frames = load_vid(CAT_VIDEO)
cat_index = 0
bum_frames = load_vid(BUM, bg_color ="green", howbig_pixels = 750, soundon = True)
bum_index = 0


#================== FACE DETECTING =================================
face_detection = mp.solutions.face_detection.FaceDetection(
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

#================ show off ;)))===================
    if cat_frames:
        cat_frame, cat_mask = cat_frames[cat_index % len(cat_frames)]
        frame = overlay_object(frame, cat_frame, cat_mask)
        cat_index += 1  #animation on the next frame every refresh    

    if bum_frames:
        bum_frame, bum_mask = bum_frames[bum_index % len(bum_frames)]
        frame = overlay_object(frame, bum_frame, bum_mask, fullscreen=True)
        bum_index += 1   
##==================================================
    cv2.imshow("camera", frame)

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
