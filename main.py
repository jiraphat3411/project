from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import tensorflow as tf
import torch
from speechbrain.inference.vocoders import HIFIGAN

app = FastAPI()

model = tf.keras.models.load_model("model.keras")

hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir="tmpdir"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    audio, sr = librosa.load(file.file, sr=16000)

    if len(audio) / sr > 10:
        return {"error": "audio longer than 10 seconds"}

    mel = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=80, 
        n_fft=1024, 
        hop_length=256, 
        fmax=8000
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - (-80.0)) / (0 - (-80.0))
    T = mel.shape[1]
    if T < 624:
        pad = 624 - T
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant",constant_values=0.0)
    elif T > 624:
        mel = mel[:, :624]
    mel = np.expand_dims(mel, axis=0)
    mel = np.expand_dims(mel, axis=-1)

    pred = model.predict(mel, verbose=0)
    # pred = pred[0, :, :, 0]
    pred = np.array(pred)[..., 0]
    pred = pred[:, :T]
    pred = pred * (0 - (-80.0)) + (-80.0)
    # pred = librosa.db_to_power(pred)
    # pred = librosa.feature.inverse.mel_to_audio(pred, 
    #     sr=16000, 
    #     n_fft=1024, 
    #     hop_length=256, 
    #     win_length=1024, 
    #     fmax=8000, 
    #     n_iter=100
    # )
    pred = np.clip(pred / 10, -11.5, 2)
    pred = torch.from_numpy(pred).float().unsqueeze(0)
    pred = hifi_gan.decode_batch(pred)
    pred = pred.squeeze().cpu().numpy()

    return {"prediction": pred.tolist()}