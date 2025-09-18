import tkinter as tk
from tkinter import ttk
import pyaudio
import wave
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===== Config =====
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "audio.wav"

# ===== Your model and labels =====
# مدل و لیبل‌ها رو جایگزین کن
model = tf.keras.models.load_model("Speech_Recognition/model.h5")  
label_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# ===== UI Functions =====
def record_and_predict():
    # --- Record Audio ---
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # --- Preprocess Audio ---
    audio = tf.io.read_file(WAVE_OUTPUT_FILENAME)
    data, sample_rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=RATE)
    data = tf.squeeze(data, axis=-1)

    spec = get_spectrogram(data)
    spec = spec[tf.newaxis, ...]

    # --- Prediction ---
    prediction = model(spec)[0].numpy()
    print(prediction)
    probs = tf.nn.softmax(prediction).numpy()
    predicted_label = label_names[tf.argmax(prediction)]

    # --- Update Chart ---
    fig.clear()
    ax = fig.add_subplot(111)
    ax.bar(label_names, probs)
    ax.set_title("Model Prediction")
    ax.set_xlabel("Words")
    ax.set_ylabel("Confidence")

    canvas.draw()

    # --- Update Label ---
    result_label.config(text=f"Predicted word: {predicted_label}", font=("Arial", 14, "bold"))

# ===== Tkinter UI =====
root = tk.Tk()
root.title("Speech Command Recognition")

# Button
start_button = ttk.Button(root, text="🎙️ Start Recording", command=record_and_predict)
start_button.pack(pady=10)

# Chart placeholder
fig = plt.Figure(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Result text
result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
