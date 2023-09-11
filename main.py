import os
import io
import math
import joblib
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import load_model
app = FastAPI()

import tensorflow as tf
# model = tf.keras.models.load_model('/kaggle/input/res-model/res_model.h5')
model = tf.keras.models.load_model('./model/res_model.h5' , custom_objects={"f1_m": 0.9998})


def noise(data, random=False, rate=0.035, threshold=0.075):
    if random:
        rate = np.random.random() * threshold
    noise_amp = rate * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data, rate=1000):
    shift_range = int(np.random.uniform(low=-5, high=5) * rate)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def chunks(data, frame_length, hop_length):
    for i in range(0, len(data), hop_length):
        yield data[i:i + frame_length]

def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def energy(data, frame_length=2048, hop_length=512):
    en = np.array([np.sum(np.power(np.abs(data[hop:hop+frame_length]), 2)) for hop in range(0, data.shape[0], hop_length)])
    return en / frame_length

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def entropy_of_energy(data, frame_length=2048, hop_length=512):
    energies = energy(data, frame_length, hop_length)
    energies /= np.sum(energies)
    entropy = -energies * np.log2(energies)
    return entropy

def spc(data, sr, frame_length=2048, hop_length=512):
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spectral_centroid)

def spc_flux(data):
    isSpectrum = data.ndim == 1
    if isSpectrum:
        data = np.expand_dims(data, axis=1)
    X = np.c_[data[:, 0], data]
    af_Delta_X = np.diff(X, 1, axis=1)
    vsf = np.sqrt((np.power(af_Delta_X, 2).sum(axis=0))) / X.shape[0]
    return np.squeeze(vsf) if isSpectrum else vsf

def spc_rollof(data, sr, frame_length=2048, hop_length=512):
    spcrollof = librosa.feature.spectral_rolloff(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spcrollof)

def chroma_stft(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.squeeze(chroma_stft.T) if not flatten else np.ravel(chroma_stft.T)

def mel_spc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    return np.squeeze(mel.T) if not flatten else np.ravel(mel.T)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_features(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset)
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    noise_data = noise(data, random=True)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))
    pitched_data = pitch(data, sample_rate, random=True)
    res3 = extract_features(pitched_data, sample_rate)
    result = np.vstack((result, res3))
    return result

def predict_emotion(audio_file):
    emos = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }
    
    features = get_features(audio_file)
    new_ip = [ele for ele in features]
    
    extracted_df = pd.DataFrame(new_ip)
    extracted_df = extracted_df.fillna(0)
    
    mm = MinMaxScaler()
    scaler = StandardScaler()
    extracted_df = mm.fit_transform(extracted_df)
    extracted_df = scaler.fit_transform(extracted_df)
    extracted_df = np.expand_dims(extracted_df, axis=2)
    
    y_pred = model.predict(extracted_df)
    y_pred = np.argmax(y_pred, axis=1)
    score = math.floor(sum(y_pred) / 3)
    return emos[score]

@app.post("/predict/")
async def predict_audio_emotion(file: UploadFile):
    if not file.filename.endswith('.wav'):
        return {"error": "Only WAV files are supported."}
    
    # Save the uploaded file temporarily
    with open('temp.wav', 'wb') as f:
        f.write(file.file.read())
    
    # Make predictions
    audio_file = 'temp.wav'
    emotion = predict_emotion(audio_file)
    
    # Clean up the temporary file
    os.remove(audio_file)
    
    return {"emotion": emotion}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
