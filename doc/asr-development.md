# run mozilla's deepspeech project
[How to Install and Use Mozilla DeepSpeech](https://progur.com/2018/02/how-to-use-mozilla-deepspeech-tutorial.html)

assume the following libraries have been installed:
- conda
- python 3.6
- tensorflow 1.10 (not needed)

## a PC without GPU
- install the deepspeech package using pip

in macosx
```
# create a brand new python 3 virtual environment with conda
conda create --name asr  python=3
conda activate asr
conda info --envs
> pip search deepspeech
deepspeech (0.3.0)         - A library for running inference on a DeepSpeech model
deepspeech-server (1.0.0)  - server for mozilla deepspeech
jeeves-deepspeech (0.1.4)  - DeepSpeech STT plugin for jeeves
deepspeech-gpu (0.3.0)     - A library for running inference on a DeepSpeech model
```

`deepspeech` is the CPU package and `deepspeech-gpu` is the gpu version.

```
pip install deepspeech
Successfully installed deepspeech-0.3.0 numpy-1.15.4
```

deepspeech v0.3 is installed, and release note is listed below
```
This is the 0.3.0 release of Deep Speech, an open speech-to-text engine. This release includes source code

v0.3.0.tar.gz

and a trained model

deepspeech-0.3.0-models.tar.gz

trained on American English which achieves an 11% word error rate on the LibriSpeech clean test corpus (models with "rounded" in their file name have rounded weights and those with a "*.pbmm" extension are memory mapped and much more memory efficient), and example audio

audio-0.3.0.tar.gz

which can be used to test the engine and checkpoint files

deepspeech-0.3.0-checkpoint.tar.gz

which can be used as the basis for further fine-tuning.
```

- download a pre-trained model released by mozilla
```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.3.0/deepspeech-0.3.0-models.tar.gz
tar -xvzf deepspeech-0.3.0-models.tar.gz
```

All we need is a sound file containing speech. DeepSpeech, however, can currently work only with signed 16-bit PCM data.

create an audio file with audacity, using a file format of `WAV(Microsoft) Signed 16-bit PCM`. A phone call audio from asterisk server is recorded as 8k mono PCM wav format.

- split the longer audio into smaller chunks based on VAD (voice activity detection) 

vad detector module webrtcvad in python
[github](https://github.com/wiseman/py-webrtcvad)

use the provided `example.py` to perform the vad operation and can modify it to support more parameters, e.g., an output directory for smaller chunks

```
pip install webrtcvad
python example.py 0 /Users/doudou/Documents/test-8k-mono-final.wav
```

- supply the chunk to deepspeech for better result
use deepspeech CLI would be sufficient for now, specify the model, audio input file and alphabet file
```
deepspeech --model=models/output_graph.pb --audio=/Users/doudou/Documents/test-audio-16k.wav --alphabet=models/alphabet.txt
```

## a PC with one GPU
install `deepspeech-gpu` first

```
pip install deepspeech-gpu
```

use deep speech project in an offline mode

# some caveat and workaround for deepspeech
## handle longer audio
deepspeech handles small chunks of audio best, so [it cannot handle longer audio directly](https://discourse.mozilla.org/t/longer-audio-files-with-deep-speech/22784)

workaround: split the audio into smaller chunks using VAD (Voice activity detection) detector, and there is a python library (py-webrtcvad) for that.

This will decrease the audio length as it remove the gap between voices. It may not be appropriate for the case that needs exact alignment between text and audio. But it should be sufficient for most cases.

## mozilla's deepspeech v0.2 support real time transcribing
[Streaming RNNs in TensorFlow](https://hacks.mozilla.org/2018/09/speech-recognition-deepspeech/)

instead of a bi-direction RNN, a uni-direction RNN is used

try this out and find out if this feature is usable in ivr scenario

Some existing robodialer blocker applications performs around 70% accuracy, and use this as a baseline.

## remove background noise from a voice recording
audacity and sox provide utility to remove background noise

if deepspeech cannot provide reasonable result, try to remove the background noise and test again.

```
Create noise file from silence + room's noise
# sox in.ext out.ext trim {start: s.ms} {duration: s.ms}
sox audio.wav noise-audio.wav trim 0 0.900
Generate a noise profile in sox:
sox noise-audio.wav -n noiseprof noise.prof
Clean the noise from the audio
sox audio.wav audio-clean.wav noisered noise.prof 0.21
According to source :

Change 0.21 to adjust the level of sensitivity in the sampling rates (I found 0.2-0.3 often provides best result).
```