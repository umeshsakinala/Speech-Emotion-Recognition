import streamlit as st
from keras.models import load_model
import numpy as np

from scipy.io import wavfile
from scipy import signal

def preprocess_speech(wav_file):
    # Read the wav file
    sample_rate, audio = wavfile.read(wav_file)

    # Normalize the audio data
    audio = audio / np.max(np.abs(audio))

    # Resample the audio to a lower sample rate
    resampled_audio = signal.resample_poly(audio, 1, sample_rate // 0.8)

    # Extract features from the audio (e.g., MFCCs, spectrogram, etc.)
    features = extract_features(resampled_audio, sample_rate)

    return features


# Load the pre-trained model
model = load_model('./model.h5')

st.title('Emotion Recognition')
st.header('This app predicts the emotion from speech')

# Upload the speech file
uploaded_file = st.file_uploader("Choose a speech file", type="wav")

if uploaded_file is not None:
    # Preprocess the speech file (you'll need to implement this)
    features = preprocess_speech(uploaded_file)
    features = np.expand_dims(features, axis=0)

    # Make a prediction
    prediction = model.predict(features)

    # Get the emotion with the highest confidence
    emotion = np.argmax(prediction)

    # Display the result
    st.write(f'The predicted emotion is: {emotion}')
