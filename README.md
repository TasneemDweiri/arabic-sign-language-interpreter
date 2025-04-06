# Arabic Sign Language Interpreter

A real-time Arabic Sign Language Interpreter that recognizes Arabic letters using webcam hand gestures. Built with *MediaPipe, **Streamlit, and a **custom-trained machine learning model*.


## 📚 Description

This project uses a *webcam* to recognize *Arabic Sign Language* gestures, converting them into Arabic letters in real-time. The system processes hand landmarks using *MediaPipe* and applies a *machine learning model* trained on a custom dataset of Arabic sign language gestures. The model is served through a *Streamlit web app*, providing an intuitive interface for easy usage.

### Key Features:
- Real-time Sign Language Recognition : Detect Arabic letters from webcam input.
- AI-Powered: Custom-trained machine learning model for gesture recognition.
- Interactive Interface: Built with Streamlit for easy-to-use, browser-based interaction.
- Supports Arabic Letters: Specifically designed for Arabic Sign Language (different from ASL).

## 🛠 Tech Stack
- Machine Learning : Scikit-learn (for training the model)
- Webcam Hand Gesture Detection: MediaPipe
- Web App Interface: Streamlit
- Data Processing: OpenCV, NumPy
- Arabic Text Rendering: arabic-reshaper, python-bidi
- Model Storage: Joblib (saved .pkl files)

## 🚀 Steps to run the app

- ***I RECCOMEND USING VIRTUAL ENVIRONMENT***
- clone this repository :
  [ git clone https://github.com/TasneemDweiri/arabic-sign-language-interpreter.git
cd arabic-sign-language-interpreter ]
- create and activate the vertual enviroment
- install all necessary dependencies (all provided in the requirements.txt)
- download model and label encoder:
[ https://huggingface.co/TasneemDweiri/arabic-sign-language-interpreter ]

⭕Make sure that the file structure is correct: you sould have [app.py, model.pkl, label_encoder.pkl, .venv/ ] all under the main folder [arabic-sign-language-interpreter] 

🌟 Now you can run the app by writing this command in the terminal [streamlit run app.py]

⭕Additional information:
- the app will use your webcam to predict the Arabic sign language gesture and display the predicted sign, make sure to have a webcam connected to your system.
