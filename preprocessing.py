'''
Place coughs in ../cough and music in ../music
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

import string
import numpy as np
import scipy.io.wavfile as scpw
import os
import random
import resampy
from retry import retry
from pedalboard import Reverb

# import sounddevice as sd

rmsStepSize = 100  # Step size for RMS analysis in milliseconds
rmsThreshold = 0.01  # 0.5  # Threshold for cutting of silence
frameLength = 1  # Size of generated audio pieces in seconds
loudestCough = -6  # the loudest possible coughing volume in dB
quietestCough = -24  # the quietest possible coughing volume in dB
globalSampleRate = 32000  # sampleRate used for the output
reverb = Reverb()
reverb.dry_level = 0.5


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


def cutIntoFrames(audioData: np.ndarray, sampleRate: int):
    sampleFrameLength = frameLengthInSamples(sampleRate, frameLength)
    audioDataLength = len(audioData) - (len(audioData) % sampleFrameLength)
    arraySplit = audioDataLength / sampleFrameLength
    audioFrames = np.split(
        audioData[0:audioDataLength], arraySplit)
    return audioFrames


'''
def playNdarray(audioData: np.ndarray, sampleRate: int):
    audioData = np.int16(audioData/np.max(np.abs(audioData)) * 32767)
    sd.play(audioData, sampleRate, blocking=True)
    return
'''


def RMS(audioData: np.ndarray):
    return np.sqrt(abs(np.mean(audioData**2)))


def dbToA(db: float):
    amplitude = 10**(db/20)
    return amplitude


def frameLengthInSamples(sampleRate: int, frameLength: int):
    frameLength = round(sampleRate * frameLength)
    return frameLength


def removeSilence(audioData: np.ndarray, sampleRate: int):
    rmsSampleStep = round(rmsStepSize*(sampleRate/1000))
    startPos = None
    endPos = None
    for i in range(0, len(audioData) - rmsSampleStep, rmsSampleStep):
        rms = RMS(audioData[i:i+rmsSampleStep])
        if startPos is None:
            if rms > rmsThreshold:
                startPos = i
        else:
            if endPos is None:
                if rms <= rmsThreshold:
                    endPos = i
    if startPos is None:
        raise ValueError('Audio Sample is too quiet.')
    if endPos is None:
        endPos = len(audioData)
    if endPos-startPos < frameLengthInSamples(sampleRate, frameLength):
        raise ValueError('Audio Sample too short.')
    else:
        audioWithoutSilence = audioData[startPos:endPos]
        return audioWithoutSilence


def addCoughToMusic(dataMusic: np.ndarray,
                    dataCough: np.ndarray,
                    sampleRate: int):
    minusThreeDb = 0.7079457843841379
    lowerLimit = dbToA(quietestCough)
    upperLimit = dbToA(loudestCough)
    dataCough = dataCough * random.uniform(lowerLimit, upperLimit)
    data = np.add(dataMusic*minusThreeDb, dataCough*minusThreeDb)
    # data = normaliseNdarray(data)
    # data = np.int16(data/np.max(np.abs(data)) * 32767)
    return data


# def addRandomVerb(data: np.ndarray, sampleRate: int):


@retry()
def processCough():
    reverb.wet_level = random.uniform(0.0, 0.5)
    reverb.room_size = random.uniform(0.1, 0.6)
    path = getRandomFile("./cough/")
    sr, data = scpw.read(path)
    data = sumToMono(data)
    # data = normaliseNdarray(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    data = reverb(data, sr)
    data = removeSilence(data, sr)
    data = cutRandomFrame(data, sr)
    if sr != globalSampleRate:
        data = resampy.resample(data, sr, globalSampleRate)
    return globalSampleRate, data


@retry()
def processMusic():
    sr, data = scpw.read(getRandomFile("./music/"))
    data = sumToMono(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    data = cutRandomFrame(data, sr)
    if sr != globalSampleRate:
        data = resampy.resample(data, sr, globalSampleRate)
    return globalSampleRate, data


def getFrame(withCough: bool, length: float):
    global frameLength
    frameLength = length
    srMusic, dataMusic = processMusic()

    if withCough:
        srCough, dataCough = processCough()
        data = addCoughToMusic(dataMusic, dataCough, srMusic)
        # vggish = waveform_to_examples(data, srMusic)
        return srMusic, data
    else:
        # vggish = waveform_to_examples(dataMusic, srMusic)
        return srMusic, dataMusic


def getTestFrames(path: string, length: float):
    global frameLength
    frameLength = length
    sr, data = scpw.read("./test_music/" + path)
    data = sumToMono(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    if sr != globalSampleRate:
        data = resampy.resample(data, sr, globalSampleRate)
    data = cutIntoFrames(data, globalSampleRate)
    return data


if __name__ == "__main__":
    while True:
        sr, data = getFrame(True, 1.0)
        # playNdarray(data, sr)
