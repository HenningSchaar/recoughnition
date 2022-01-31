'''
Place coughs in ./cough and music in ./music
This is the place for creating audio chunks with corresponding data about if
and where there are coughs in the resulting audio chunks.

The resulting output is an array of objects. The objects need two variables:
the audio-waveform data and the position Data of the cough, given in frames.

For preparing the cough data in Terminal on cough folder run:

"for f in *.ogg
      ffmpeg -i $f -c:a pcm_f32le $f.wav
  end"
  
If this file is run as main you will hear randomly generated frames.
To use this as an import use:

"from preprocessing import getFrame"

The getFrame function will return sampleRate: int, audioData: np.ndarray.

'''

import numpy as np
import scipy.io.wavfile as scpw
import sounddevice as sd
import os
import random
from retry import retry
rmsStepSize = 100  # Step size for RMS analysis in milliseconds
rmsThreshold = 0.001  # 0.5  # Threshold for cutting of silence
frameLength = 1000  # Size of generated audio pieces in seconds
coughScaling = 1  # relation in amplitude between cough and music


def getRandomFile(directory: str):
    chosenFile = random.choice(os.listdir(directory))
    # print(chosenFile)
    chosenPath = directory + chosenFile
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


def cutRandomFrame(audioData: np.ndarray, sampleRate: int):
    sampleFrameLength = frameLengthInSamples(sampleRate, frameLength)
    range = len(audioData) - sampleFrameLength
    startPos = random.randrange(range)
    endPos = startPos + sampleFrameLength
    randomSecondAudio = audioData[startPos:endPos]
    # print(f"start: {startPos} length: {len(randomSecondAudio)}")
    return randomSecondAudio


def playNdarray(audioData: np.ndarray, sampleRate: int):
    # audioData = np.int16(audioData/np.max(np.abs(audioData)) * 32767)
    sd.play(audioData, sampleRate, blocking=True)
    return


def RMS(audioData: np.ndarray):
    return np.sqrt(abs(np.mean(audioData**2)))


def frameLengthInSamples(sampleRate: int, frameLength: int):
    frameLength = round((sampleRate*1000)/frameLength)
    return frameLength


def removeSilence(audioData: np.ndarray, sampleRate: int):
    rmsSampleStep = round(rmsStepSize*(sampleRate/1000))
    startPos = None
    endPos = None
    for i in range(0, len(audioData) - rmsSampleStep, rmsSampleStep):
        rms = RMS(audioData[i:i+rmsSampleStep])
        if startPos is None:
            if rms >= rmsThreshold:
                startPos = i
        else:
            if endPos is None:
                if rms <= rmsThreshold:
                    endPos = i
    if endPos-startPos < frameLengthInSamples(sampleRate, frameLength):
        raise ValueError('Audio Sample too short.')
    else:
        audioWithoutSilence = audioData[startPos:endPos]
        return audioWithoutSilence


def addCoughToMusic(dataMusic: np.ndarray, dataCough: np.ndarray, sampleRate: int):
    minusThreeDb = 0.7079457843841379
    dataCough = dataCough * coughScaling
    data = np.add(dataMusic*minusThreeDb, dataCough*minusThreeDb)
    data = normaliseNdarray(data)
    # data = np.int16(data/np.max(np.abs(data)) * 32767)
    return data


@retry()
def processCough():
    sr, data = scpw.read(getRandomFile("./cough/"))
    data = sumToMono(data)
    data = normaliseNdarray(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    data = removeSilence(data, sr)
    data = cutRandomFrame(data, sr)
    return sr, data


@retry()
def processMusic():
    sr, data = scpw.read(getRandomFile("./music/"))
    data = sumToMono(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    data = cutRandomFrame(data, sr)
    return sr, data


def getFrame():
    srCough, dataCough = processCough()
    srMusic, dataMusic = processMusic()
    if srMusic != srCough:
        # A function for matching the sample rates
        # has to be written in the future.
        raise ValueError('Sample rates of music and cough do not match.')
    data = addCoughToMusic(dataMusic, dataCough, srMusic)
    return srMusic, data


if __name__ == "__main__":
    # Load random audio file from cough folder. (.wav)
    while True:
        sr, data = getFrame()
        playNdarray(data, sr)
        '''
        sr, data = processCough()
        playNdarray(data, sr)
        sr, data = processMusic()
        playNdarray(data, sr)
        '''
