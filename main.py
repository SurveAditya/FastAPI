import os
import io
import math
import joblib
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pymongo import MongoClient
import os
import aiofiles
from datetime import datetime
# import pymongo





from keras.models import load_model
app = FastAPI()

import tensorflow as tf
# model = tf.keras.models.load_model('/kaggle/input/res-model/res_model.h5')
model = tf.keras.models.load_model('./model/third.h5')

# Setting up mongodb atlas connection

client = MongoClient("mongodb+srv://admin:admin@cluster0.qs1hl6j.mongodb.net")
db = client["sih2023"]


def mfcc_extractor(audio,sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

def zcr_extractor(audio, sample_rate):
    zcr_features = librosa.feature.zero_crossing_rate(y=audio)
    zcr_scaled_features = np.mean(zcr_features.T,axis=0)
    
    return zcr_scaled_features

def srf_extractor(audio, sample_rate):
    S, phase = librosa.magphase(librosa.stft(audio))
    srf_features = librosa.feature.spectral_rolloff(S=S, sr=sample_rate)
    srf_scaled_features = np.mean(srf_features.T,axis=0)
    
    return srf_scaled_features

def flux_extractor(audio, sample_rate):
    flux_features = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    flux_scaled_features = np.mean(flux_features.T,axis=0)
    
    return flux_scaled_features

def chroma_extractor(audio,sample_rate):
    chroma_features = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled_features = np.mean(chroma_features.T,axis=0)
    
    return chroma_scaled_features

def single_val(data):
    return data[0]

# audio, sample_rate = librosa.load(filename) 
def ip(file, third_model):
    emos = {
        0: "angry", 
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }
    data, sr = librosa.load(file)
    extracted_features = []
    mfcc = mfcc_extractor(data,sr)
    zcr = zcr_extractor(data,sr)
    srf = srf_extractor(data,sr)
    flux = flux_extractor(data,sr)
    chroma = chroma_extractor(data,sr)
    extracted_features.append([mfcc,zcr,srf,flux,chroma])
    new_ip=pd.DataFrame(extracted_features,columns=['mfcc','zcr','srf','flux', 'chroma'])
    new_ip['zcr'] = new_ip['zcr'].apply(single_val)
    new_ip['srf'] = new_ip['srf'].apply(single_val)
    ss = StandardScaler()
    new_ip[['zcr', 'srf', 'flux']] = ss.fit_transform(new_ip[['zcr', 'srf', 'flux']])
    new_ip_mfcc=np.array(new_ip['mfcc'].tolist())
    new_ip_zcr=np.array(new_ip['zcr'].tolist()).reshape(-1,1)
    new_ip_srf=np.array(new_ip['srf'].tolist()).reshape(-1,1)
    new_ip_flux=np.array(new_ip['flux'].tolist()).reshape(-1,1)
    new_ip_chroma=np.array(new_ip['chroma'].tolist())
    new_ip_X = np.concatenate((new_ip_mfcc, new_ip_zcr, new_ip_srf, new_ip_flux, new_ip_chroma), axis=1)
    # print(new_ip_X.shape)
    #print(mfccs_scaled_features)
    #mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    #print(mfccs_scaled_features)
    #print(mfccs_scaled_features.shape)
    predicted_label=third_model.predict(new_ip_X)
    predicted_label = predicted_label.reshape(-1)
    print(predicted_label)
    ind = np.argmax(predicted_label)
    print(np.argmax(predicted_label))
    return emos[ind]


@app.post("/predict/")
async def predict_audio_emotion(file: UploadFile):
    try:
        # Create a temporary file to save the uploaded data
        temp_filename = 'temp.wav'
        async with aiofiles.open(temp_filename, 'wb') as temp_file:
            # Iterate over the file content in chunks and write it to the temporary file
            while chunk := await file.read(1024):
                await temp_file.write(chunk)

        # Make predictions
        emotion = ip(temp_filename, model)

        # Clean up the temporary file
        os.remove(temp_filename)

        # Return the emotion prediction
        return {"emotion": emotion}
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {str(e)}")
        return {"error": "An error occurred during audio processing."}

    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
