import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

# Arabic map
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

# Load trained model and encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Function to draw Arabic text using Pillow
def draw_arabic_text(frame, text, position):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Convert OpenCV image to PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Use an Arabic-compatible TTF font (adjust path as needed)
    font = ImageFont.truetype("arial.ttf", 40)  # Try "arial.ttf" or "NotoNaskhArabic-Regular.ttf"

    # Draw text
    draw.text(position, bidi_text, font=font, fill=(0, 255, 0))

    # Convert back to OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Webcam start
cap = cv2.VideoCapture(0)
print("Starting real-time Arabic sign language interpreter (press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                label = label_encoder.inverse_transform([prediction])[0]
                arabic_letter = EnAr.get(label, label)

                # Draw Arabic letter
                frame = draw_arabic_text(frame, arabic_letter, (10, 50))

    cv2.imshow("Arabic Sign Language Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()