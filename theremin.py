import cv2
import numpy as np
import sounddevice as sd
from ultralytics import YOLO
import threading

# Audio Settings
SAMPLE_RATE = 44100
BLOCK_SIZE = 512

current_frequency = 440.0
current_volume = 0.0
phase = 0.0

target_frequency = 440.0
target_volume = 0.0

smoothed_frequency = 440.0
smoothed_volume = 0.0

SMOOTHING = 0.001

audio_lock = threading.Lock()

NOTE_SEQUENCE = [
    "Bb3","C4","D4","Eb4","F4","G4","A4",
    "Bb4","C5","D5","Eb5","F5","G5","A5","Bb5"
]

NOTE_FREQUENCIES = {
    "Bb3": 233.08,
    "C4": 261.63,
    "D4": 293.66,
    "Eb4": 311.13,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "Bb4": 466.16,
    "C5": 523.25,
    "D5": 587.33,
    "Eb5": 622.25,
    "F5": 698.46,
    "G5": 783.99,
    "A5": 880.00,
    "Bb5": 932.33,
}

def map_height_to_note(y, frame_height):
    normalized = 1.0 - (y / frame_height)
    normalized = np.clip(normalized, 0, 1)
    index = int(normalized * (len(NOTE_SEQUENCE) - 1))
    return NOTE_SEQUENCE[index]

def map_height_to_volume(y, frame_height):
    normalized = 1.0 - (y / frame_height)
    return float(np.clip(normalized, 0, 1))

# Audio Callback
def audio_callback(outdata, frames, time, status):
    global phase
    global smoothed_frequency, smoothed_volume

    if status:
        print(status)

    out = np.zeros(frames)

    with audio_lock:
        tf = target_frequency
        tv = target_volume

    for i in range(frames):
        # Smooth frequency and volume toward targets
        smoothed_frequency += (tf - smoothed_frequency) * SMOOTHING
        smoothed_volume += (tv - smoothed_volume) * SMOOTHING

        # Increment phase
        phase += 2 * np.pi * smoothed_frequency / SAMPLE_RATE
        if phase > 2 * np.pi:
            phase -= 2 * np.pi

        out[i] = smoothed_volume * np.sin(phase)

    outdata[:] = out.reshape(-1, 1)

# Start audio stream
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    callback=audio_callback
)
stream.start()


# yummy yolo model for skeleton

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

MIN_PERSON_CONF = 0.5

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    display = np.ones_like(frame) * 255

    results = model(frame)

    for result in results:
        if result.keypoints is None:
            continue

        keypoints = result.keypoints.xy
        confidences = result.keypoints.conf

        best_person_id = None
        best_score = 0

        for person_id in range(len(keypoints)):
            avg_conf = confidences[person_id].mean()
            if avg_conf >= MIN_PERSON_CONF and avg_conf > best_score:
                best_score = avg_conf
                best_person_id = person_id

        if best_person_id is None:
            continue

        person = keypoints[best_person_id]
        person_conf = confidences[best_person_id]

        # skeleton
        for (kp1, kp2) in SKELETON:
            if person_conf[kp1] > 0.5 and person_conf[kp2] > 0.5:
                x1, y1 = person[kp1]
                x2, y2 = person[kp2]
                cv2.line(display,
                         (int(x1), int(y1)),
                         (int(x2), int(y2)),
                         (0, 0, 0), 2)

        # keypoints
        for kp_id, (x, y) in enumerate(person):
            if kp_id >= 5 and person_conf[kp_id] > 0.5:
                cv2.circle(display,
                           (int(x), int(y)),
                           4, (0, 0, 0), -1)

        # LEFT WRIST for pitch
        if person_conf[9] > 0.5:
            note = map_height_to_note(person[9][1], h)
            freq = NOTE_FREQUENCIES[note]

            with audio_lock:
                target_frequency = freq

            cv2.putText(display,
                        f"Pitch: {note}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 0), 3)

        # RIGHT WRIST for vol
        if person_conf[10] > 0.5:
            volume = map_height_to_volume(person[10][1], h)

            with audio_lock:
                target_volume = volume

            cv2.putText(display,
                        f"Volume: {volume:.2f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 0), 3)

    cv2.imshow("Digital Theremin", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()