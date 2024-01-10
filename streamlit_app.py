import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import pickle
import librosa
import os
import speech_recognition as sr
import pyautogui
import time

model_file_path = 'XGBoost_finetuned.sav'
loaded_model = pickle.load(open(model_file_path, 'rb'))

# Streamlit UI
st.title('Trigger Word Detection')

# Initialize session state
if 'recording_done' not in st.session_state:
    st.session_state.recording_done = False
    st.session_state.speech_text = ""

recognizer = sr.Recognizer()

# Record audio from the microphone
is_recording = st.button("Start Recording")

def mfcc_features(signal, sample_rate):
    return np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T, axis=0).tolist()
p = [0,1,2]

def compute_zero_crossing_rate(audio):
    return librosa.feature.zero_crossing_rate(audio)[0]

def compute_spectral_centroid(audio, sample_rate):
    # Compute the spectrogram
    spectrogram = np.abs(librosa.stft(audio))

    # Compute the spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(S=spectrogram, sr=sample_rate)[0]

    return spectral_centroid

def compute_rms_energy(audio):
    # Compute the root mean square (RMS) energy
    rms_energy = librosa.feature.rms(y=audio)[0]


# Function to capture audio and make predictions
def capture_audio():
    while not st.session_state.recording_done:
        audio_data = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()

        temp_file_path = "temp_audio.wav"
        sf.write(temp_file_path, audio_data.flatten(), 16000)

        try:
            with sr.AudioFile(temp_file_path) as source:
                audio_data = recognizer.record(source)
                st.session_state.speech_text = recognizer.recognize_google(audio_data, language="en-US")
        except Exception as ex:
            st.error(f"No Trigger - retry: {ex}")

        signal, rate = sf.read(temp_file_path)

        if isinstance(signal, np.ndarray) and signal.ndim == 1:
            mfcc = mfcc_features(signal, rate)
            czcr = compute_zero_crossing_rate(signal)
            csc = compute_spectral_centroid(signal, rate)
            rms_energy = compute_rms_energy(signal)

            mfcc = np.array(mfcc)
            czcr = np.array(czcr)
            csc = np.array(csc)
            rms_energy = np.array(rms_energy)

            mfcc_reshaped = mfcc.reshape(1, -1)
            czcr_reshaped = czcr.reshape(1, -1)
            czcr_reshaped = czcr_reshaped[:,0:32]
            csc_reshaped = csc.reshape(1, -1)
            csc_reshaped = csc_reshaped[:,0:32]
            rms_energy_reshaped = rms_energy.reshape(1, -1)
            rms_energy_reshaped = rms_energy_reshaped[:,0:32]

            mfcc_flat = mfcc_reshaped.flatten()
            czcr_flat = czcr_reshaped.flatten()
            csc_flat = csc_reshaped.flatten()
            rms_energy_flat = rms_energy_reshaped.flatten()
            #print(len(mfcc_flat))
            #print(len(czcr_flat))
            #print(len(csc_flat))
            #print(len(rms_energy_flat))

            # Create a single list for prediction
            input_data = []
            input_data.extend(mfcc_flat.tolist())
            input_data.extend(czcr_flat.tolist())
            input_data.extend(csc_flat.tolist())
            input_data.extend(rms_energy_flat.tolist())

            # Predict using the reshaped mfcc
            input_data_reshaped = np.array(input_data).reshape(1, -1)
            y_pred = loaded_model.predict(input_data_reshaped)

            # Map predicted labels back to original class labels
            #predicted_class = le.inverse_transform(y_pred)
            print("Predicted Class:", y_pred)

            if st.session_state.speech_text.lower() == "door open" and y_pred[0] == 1:
                st.success(f'Trigger Door Open 🚪🔓✔')
                st.session_state.recording_done = True
            elif st.session_state.speech_text.lower() == "door close" and y_pred[0] == 1:
                st.success(f'Trigger Door Close 🚪🔒✔')
                st.session_state.recording_done = True
            elif st.session_state.speech_text.lower() == "door stop" and y_pred[0] == 1:
                st.success(f'Trigger Door Stop 🚪🚫')
                st.session_state.recording_done = True
            else:
                st.success('Trigger Word Not Detected ⛔⛔⛔ /n moye moye!!')

        os.remove(temp_file_path)

# Start capturing audio when the website is loaded
if is_recording:
    capture_audio()

# Example: Make prediction using the loaded model and speech recognition result
if st.session_state.recording_done:
    st.success("Done")
    time.sleep(3)
    pyautogui.hotkey("f5")
