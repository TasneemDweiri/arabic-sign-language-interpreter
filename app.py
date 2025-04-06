import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import arabic_reshaper
from bidi.algorithm import get_display
from huggingface_hub import hf_hub_download
from PIL import ImageFont, ImageDraw, Image

model_path = hf_hub_download(repo_id="TasneemDweiri/arabic-sign-language-interpreter", filename="model.pkl")
label_encoder_path = hf_hub_download(repo_id="TasneemDweiri/arabic-sign-language-interpreter", filename="label_encoder.pkl")

# Load model and label encoder
model = joblib.load("model_path")
label_encoder = joblib.load("label_encoder_path")

# Arabic letters mapping
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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Draw Arabic text on image
def draw_arabic(frame, text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), bidi_text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# App UI
st.title("Arabic Sign Language Interpreter")
st.write("Show your hand gesture to the webcam and get the Arabic letter in real-time!")

run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to read from webcam.")
            break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                label = label_encoder.inverse_transform([prediction])[0]
                arabic_letter = EnAr.get(label, label)
                frame = draw_arabic(frame, arabic_letter)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
