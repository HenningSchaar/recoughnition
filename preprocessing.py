'''
This is the place for creating audio chunks with corresponding data about if
and where there are coughs in the resulting audio chunks.

The resulting output is an array of objects. The objects need two variables:
the audio-waveform data and the position Data of the cough, given in frames.

For preparing the cough data in Terminal on cough folder run:

for f in *.ogg
      ffmpeg -i $f -c:a pcm_f32le $f.wav
  end

'''
from email.mime import audio
import numpy as np
import scipy.io.wavfile as scpw
import sounddevice as sd
import os
import random


def getRandomCough():
    chosenFile = random.choice(os.listdir("./cough"))
    chosenPath = "./cough/" + chosenFile
    return chosenPath


def sumToMono(audioData: np.ndarray):
    if len(audioData.shape) == 2:
        data = np.sum(audioData, axis=1)
        return data
    else:
        return audioData


def normaliseNdarray(audioData: np.ndarray):
    # get biggest absolute value for normalisation
    absmax = max(audioData, key=abs)
    audioData = np.divide(audioData, absmax)  # normalise values
    return audioData


def playNdarray(audioData: np.ndarray, sampleRate: int):
    sd.play(audioData, sampleRate, blocking=True)
    return


if __name__ == "__main__":
    # Load random audio file from cough folder. (.wav)
    sr, data = scpw.read(getRandomCough())
    data = sumToMono(data)
    data = normaliseNdarray(data)
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    playNdarray(data, sr)
