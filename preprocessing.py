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
import numpy as np
import soundfile as sf
import os
import random


def getRandomCough():
    return "./cough/" + random.choice(os.listdir("./cough"))


if __name__ == "__main__":
    # Load random audio file from cough folder. (.wav)
    data, samplerate = sf.read(getRandomCough())
    data = np.divide(data, data.max())  # normalise the audio file
