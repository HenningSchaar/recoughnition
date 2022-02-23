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

import numpy as np
import scipy.io.wavfile as scpw
import os
import random
import scipy
import resampy
from retry import retry

from recoughnition import vggish_params
from recoughnition import mel_features
rmsStepSize = 100  # Step size for RMS analysis in milliseconds
rmsThreshold = 0.001  # 0.5  # Threshold for cutting of silence
frameLength = 1  # Size of generated audio pieces in seconds
coughScaling = 0.5  # relation in amplitude between cough and music


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.
    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.
    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands]
      which represents a sequence of examples, each of which contains a patch
      of log mel spectrogram, covering num_frames frames of audio and num_bands
      mel frequency bands, where the frame length is
      vggish_params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    return log_mel_examples


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
    # sd.play(audioData, sampleRate, blocking=True)
    return


def RMS(audioData: np.ndarray):
    return np.sqrt(abs(np.mean(audioData**2)))


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


def addCoughToMusic(dataMusic: np.ndarray,
                    dataCough: np.ndarray,
                    sampleRate: int):
    minusThreeDb = 0.7079457843841379
    dataCough = dataCough * coughScaling
    data = np.add(dataMusic*minusThreeDb, dataCough*minusThreeDb)
    #data = normaliseNdarray(data)
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
    if sr != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sr, vggish_params.SAMPLE_RATE)
    return vggish_params.SAMPLE_RATE, data


@retry()
def processMusic():
    sr, data = scpw.read(getRandomFile("./music/"))
    data = sumToMono(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    data = cutRandomFrame(data, sr)
    if sr != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sr, vggish_params.SAMPLE_RATE)
    return vggish_params.SAMPLE_RATE, data


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
'''
def cutTestFrame(audioData: np.ndarray, sampleRate: int, frame_number:int):
    randomSecondAudio = audioData[sampleRate*frame_number:sampleRate*(frame_number+1)]
    # print(f"start: {startPos} length: {len(randomSecondAudio)}")
    return randomSecondAudio

def processMusic2(filename, frame_number):
    sr, data = scpw.read(filename)
    data = sumToMono(data)
    data = normaliseNdarray(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    data = cutTestFrame(data, sr, frame_number)
    if sr != vggish_params.SAMPLE_RATE:
            data = resampy.resample(data, sr, vggish_params.SAMPLE_RATE)
    return vggish_params.SAMPLE_RATE, data

def getFrame2(filename, frame_number):
    srMusic, dataMusic = processMusic2(filename, frame_number)
    return srMusic, dataMusic
'''
if __name__ == "__main__":
    # Load random audio file from cough folder. (.wav)
    while True:
        sr, data, vggish = getFrame(False, 0.97)
        print(vggish)
        sr, data, vggish = getFrame(True, 0.97)
        print(vggish)
