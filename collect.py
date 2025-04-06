import os
import cv2
import mediapipe as mp
import csv

# Dataset and output paths
DATASET_DIR = "Dataset"  # Change if your folder name is different
CSV_OUTPUT = 'data/gestures.csv'

# Mapping from folder names to Arabic letters
EnAr = {
    "Alef": "أ", "Beh": "ب", "Teh": "ت", "Theh": "ث",
    "Jeem": "ج", "Hah": "ح", "Khah": "خ", "Dal": "د",
    "Thal": "ذ", "Reh": "ر", "Zain": "ز", "Seen": "س",
    "Sheen": "ش", "Sad": "ص", "Dad": "ض", "Tah": "ط",
    "Zah": "ظ", "Ain": "ع", "Ghain": "غ", "Feh": "ف",
    "Qaf": "ق", "Kaf": "ك", "Lam": "ل", "Meem": "م",
    "Noon": "ن", "Heh": "ه", "Waw": "و", "Yeh": "ي",
    "Laa": "لا", "Teh_Marbuta": "ة", "Al": "ال"
}

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Write to gestures.csv
with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    total = 0

    for folder in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(label_path) or folder not in EnAr:
            continue

        arabic_label = EnAr[folder]

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks.append(arabic_label)
                    writer.writerow(landmarks)
                    total += 1
                    break

print(f"Finished! Extracted {total} labeled samples into {CSV_OUTPUT}")