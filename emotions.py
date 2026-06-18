import cv2
import torch
from PIL import Image
from transformers import pipeline
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import pygame

SAVED_FACES_DIR = "saved_faces"
os.makedirs(SAVED_FACES_DIR, exist_ok=True)

# ++++++ SMOOTHING CONFIG ++++++
# blizej zera -> gładsze przejścia
# bliżej jedynki -> gwałtowniejsze skoki
EMA_ALPHA = 0.2
# ++++++++++++++++++++++++++++++
smoothed = {
    "angry":    0.0,
    "boredom":  0.0,
    "disgust":  0.0,
    "fear":     0.0,
    "happy":    0.0,
    "neutral":  0.0,
    "sad":      0.0,
    "surprise": 0.0,
}

# ++++++ EMOTION COLORS ++++++
# wybor stalych kolorow dla kazdej z emocji
EMOTION_COLORS = {
    "angry":    "#e74c3c",   # czerwony
    "boredom":  "#95a5a6",   # szary
    "disgust":  "#8e44ad",   # fiolet
    "fear":     "#2c3e50",   # granatowy
    "happy":    "#f1c40f",   # żółty
    "neutral":  "#3498db",   # niebieski
    "sad":      "#1abc9c",   # turkus
    "surprise": "#e67e22",   # pomarańczowy
}

# Mapowanie emocji -> numer poziomu na wykresie schodkowym
EMOTIONS = list(smoothed.keys())
EMOTION_LEVELS = {emo: i for i, emo in enumerate(EMOTIONS)}

# -------------------------------------------------------------------------------
# Ładowanie modelu ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("image-classification", model="vit-emotion-final", device=device)
print("Using device:", "cuda" if device >= 0 else "cpu")

# -------------------------------------------------------------------------------
# Funkcja do przewidywania emocji +++++++++++++++++++++++++++++++++++++++++++++++
def predict_emotion(face_img):
    image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    results = pipe(image, top_k=None)
    return {r["label"]: float(r["score"]) for r in results}


# --------------------------------------------------------------------------------
# Funkcje do nakładania gifów na klatke ++++++++++++++++++++++++++++++++++++++++++
def load_vid(path, bg_color="black", howbig_pixels=450):
    color_ranges = {
        "black": (np.array([0,   0,   0]),   np.array([180, 255,  40])),
        "green": (np.array([40,  100, 100]), np.array([57,  221, 255])),
        "white": (np.array([0,   0,   180]), np.array([180,  40, 255])),
    }
    cap_v = cv2.VideoCapture(path)
    frames = []
    lower, upper = color_ranges.get(bg_color, color_ranges["black"])
    while True:
        ret, frame = cap_v.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_bg = cv2.inRange(hsv, lower, upper)
        mask_obj = cv2.bitwise_not(mask_bg)
        new_w = howbig_pixels
        new_h = int(frame.shape[0] * howbig_pixels / frame.shape[1])
        frames.append((cv2.resize(frame, (new_w, new_h)),
                       cv2.resize(mask_obj, (new_w, new_h))))
    cap_v.release()
    return frames

def overlay_object(background, obj_frame, obj_mask, margin=10, fullscreen=False):
    bg_h, bg_w = background.shape[:2]
    if fullscreen:
        obj_frame = cv2.resize(obj_frame, (bg_w, bg_h))
        obj_mask  = cv2.resize(obj_mask,  (bg_w, bg_h))
        x_offset, y_offset = 0, 0
    else:
        obj_h, obj_w = obj_frame.shape[:2]
        x_offset = margin
        y_offset = bg_h - obj_h - margin
        if y_offset < 0 or x_offset + obj_w > bg_w:
            return background
    obj_h, obj_w = obj_frame.shape[:2]
    roi = background[y_offset:y_offset + obj_h, x_offset:x_offset + obj_w]
    alpha = obj_mask.astype(float) / 255.0
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    blended = obj_frame.astype(float) * alpha_3ch + roi.astype(float) * (1 - alpha_3ch)
    background[y_offset:y_offset + obj_h, x_offset:x_offset + obj_w] = blended.astype(np.uint8)
    return background

# --------------------------------------------------------------------------------
# Ładowanie gifów i dźwięku ++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("CAT loading...")
cat_frames = load_vid(r"media\cat_vid.mp4", bg_color="black", howbig_pixels=450)
print(f"{len(cat_frames)} cat frames loaded.")
cat_index = 0

print("MLG loading...")
mlg_frames = load_vid(r"media\mlg.mp4", bg_color="green", howbig_pixels=750)
print(f"{len(mlg_frames)} MLG frames loaded.")
mlg_index = 0

pygame.mixer.init()
mlg_sound = pygame.mixer.Sound(r"media\mlg_classic.mp3")
mlg_sound_playing = False

# --------------------------------------------------------------------------------
# Inicjalizujemy detektor twarzy i włączamy kamere +++++++++++++++++++++++++++++++
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(1)

# --------------------------------------------------------------------------------
# Ustawianie plotowania ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
start_time = time.time()

history = {emo: [] for emo in EMOTIONS}
timestamps = []

# Dane do wykresu schodkowego (dominująca emocja)
dominant_times  = []   # punkty czasowe gdzie zmienia sie emocja
dominant_levels = []   # poziomy emocji na wykresie
dominant_colors = []   # kolor każdego kawałka

# Czas trwania każdej emocji (do słupkowego)
emotion_duration = {emo: 0.0 for emo in EMOTIONS}
last_emotion     = None
last_t           = 0.0

plt.ion()

# ------------------------------------------------------------------------------------
# Wykresy: góra - pewność sieci, środek - domiujaca emocja, dół - procentowy wkład emocji w czasie
fig = plt.figure(figsize=(12, 10))
gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1.5], hspace=0.45)

ax_prob  = fig.add_subplot(gs[0])   # wykres pewności co do danych emocji
ax_step  = fig.add_subplot(gs[1])   # dominujaca emocja
ax_bar   = fig.add_subplot(gs[2])   # udział procentowy

# --- ax_prob ---
prob_lines = {}
for emo in EMOTIONS:
    line, = ax_prob.plot([], [], label=emo, color=EMOTION_COLORS[emo], linewidth=1.5)
    prob_lines[emo] = line
ax_prob.set_ylim(0, 1)
ax_prob.set_ylabel("Probability")
ax_prob.set_title("Model confidence per emotion")
ax_prob.legend(loc="upper left", fontsize=7, ncol=2)
ax_prob.grid(True, alpha=0.3)

# --- ax_step ---
ax_step.set_ylim(-0.5, len(EMOTIONS) - 0.5)
ax_step.set_yticks(range(len(EMOTIONS)))
ax_step.set_yticklabels(EMOTIONS, fontsize=9)
ax_step.set_ylabel("Dominant emotion")
ax_step.set_title("Dominant emotion over time")
ax_step.grid(True, alpha=0.3, axis='x')

step_line, = ax_step.plot([], [], color="gray", linewidth=2, zorder=1)
step_dot   = ax_step.scatter([], [], s=60, zorder=3)  # bieżący punkt
step_label = ax_step.text(0, 0, "", fontsize=10, fontweight="bold",
                          va="bottom", ha="left", color="white",
                          bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

# --- ax_bar ---
bar_colors = [EMOTION_COLORS[emo] for emo in EMOTIONS]
bars = ax_bar.bar(EMOTIONS, [0]*len(EMOTIONS), color=bar_colors)
ax_bar.set_ylim(0, 100)
ax_bar.set_ylabel("Time share (%)")
ax_bar.set_title("Emotion time distribution")
ax_bar.tick_params(axis='x', labelsize=8)
ax_bar.grid(True, alpha=0.3, axis='y')



# --------------------------------------------------------------------------------
# Funkcja do odświeżania wszystkich wykresów +++++++++++++++++++++++++++++++++++++
def update_plots(t, current_emotion):
    """Odświeża wszystkie trzy wykresy."""
    global last_emotion, last_t

    # --- wykres pewności ---
    for emo in EMOTIONS:
        prob_lines[emo].set_xdata(timestamps)
        prob_lines[emo].set_ydata(history[emo])
    ax_prob.relim()
    ax_prob.autoscale_view(scalex=True, scaley=False)

    # --- wykres dominujacej emocji ---
    if current_emotion is not None:
        level = EMOTION_LEVELS[current_emotion]
        color = EMOTION_COLORS[current_emotion]

        if last_emotion is None:
            # pierwszy punkt traktowany inaczej
            dominant_times.extend([t, t])
            dominant_levels.extend([level, level])
        elif current_emotion != last_emotion:
            # przeskok emocji - SKOK
            dominant_times.extend([t, t])
            dominant_levels.extend([EMOTION_LEVELS[last_emotion], level])
        else:
            # bez zmian - ta sama emocja
            dominant_times.append(t)
            dominant_levels.append(level)

        step_line.set_xdata(dominant_times)
        step_line.set_ydata(dominant_levels)

        # kropka ruszajaca sie na przodzie wykresu
        step_dot.set_offsets([[t, level]])
        step_dot.set_color(color)

        # nazwa emocji przy kropce
        step_label.set_position((t, level + 0.15))
        step_label.set_text(current_emotion)
        step_label.set_color(color)

        ax_step.relim()
        ax_step.autoscale_view(scalex=True, scaley=False)
        ax_step.set_ylim(-0.5, len(EMOTIONS) - 0.5)

        if last_emotion is not None:
            emotion_duration[last_emotion] += t - last_t
        last_emotion = current_emotion
        last_t = t

    # --- wykres udziału procentowego   ---
    total = sum(emotion_duration.values())
    if total > 0:
        for bar, emo in zip(bars, EMOTIONS):
            bar.set_height(100 * emotion_duration[emo] / total)
    ax_bar.set_ylim(0, 100)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ------------------ GŁÓWNA PĘTLA ----------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Starting webcam... Press Q to quit")
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) #lustrzane odbicie, zeby naturalnie było

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    current_emotion = None

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        scores = predict_emotion(face_crop)

        t = time.time() - start_time
        timestamps.append(t)

        for emo in EMOTIONS:
            raw = scores.get(emo, 0.0)
            smoothed[emo] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * smoothed[emo]
            history[emo].append(smoothed[emo])

        current_emotion = max(smoothed, key=smoothed.get)
        confidence = smoothed[current_emotion]
        print(current_emotion)

        # prostokąt zmienia kolor zależnie od wykrytej emocji
        hex_c = EMOTION_COLORS[current_emotion].lstrip("#")
        bgr   = tuple(int(hex_c[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, 2)
        cv2.putText(frame, f"{current_emotion} ({confidence:.2f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)

    # -----------------------------------------------------------------------
    # logika wyświetlania materiałów od aktulnej emocji +++++++++++++++++++++
    if current_emotion == "happy" and cat_frames:
        cf, cm = cat_frames[cat_index % len(cat_frames)]
        frame = overlay_object(frame, cf, cm, fullscreen=False)
        cat_index += 1
        if mlg_sound_playing:
            mlg_sound.stop()
            mlg_sound_playing = False

    elif current_emotion == "boredom" and mlg_frames:
        mf, mm = mlg_frames[mlg_index % len(mlg_frames)]
        frame = overlay_object(frame, mf, mm, fullscreen=True)
        mlg_index += 1
        if not mlg_sound_playing:
            mlg_sound.play(loops=-1)
            mlg_sound_playing = True

    else:
        if mlg_sound_playing:
            mlg_sound.stop()
            mlg_sound_playing = False
    # ===============================================

    # -----------------------------------------------------------------------
    # odświeżanie wykresów ++++++++++++++++++++++++++++++++++++++++++++++++++
    if timestamps:
        update_plots(timestamps[-1], current_emotion)

    cv2.imshow("Emotion Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) > 0:
            ts = int(time.time())
            for i, (x, y, w, h) in enumerate(faces):
                fc = frame[y:y+h, x:x+w].copy()
                sc = predict_emotion(fc)
                re = max(sc, key=sc.get)
                rc = sc[re]
                fn = f"face_{i}_{re}_{rc:.2f}_{ts}.jpg"
                cv2.imwrite(os.path.join(SAVED_FACES_DIR, fn), fc)
                print(f"Saved: {fn}")

    elif key == ord('q'):
        # zapis do pliku CSV całej serii czasowej
        df = pd.DataFrame(history)
        df["time"] = timestamps
        csv_path = os.path.join(SAVED_FACES_DIR, "emotion_history.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # Zapisz oba wykresy jako osobne pliki
        fig.savefig(os.path.join(SAVED_FACES_DIR, "emotion_plot_full.png"), dpi=300, bbox_inches="tight")
        print("Saved full plot.")

        break

# zamykanie wszystkiego
mlg_sound.stop()
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()