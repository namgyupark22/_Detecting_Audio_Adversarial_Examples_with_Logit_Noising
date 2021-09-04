## Detecting_Audio_Adversarial_Examples_With_Logit_Noising

This repository contains code of ACSAC 2021 paper "Detecting Audio Adversarial Examples with Logit Noising".

In thi repository, we upload Det_DeepSpeech.py, logit_analysis_ACSAC.ipyb, and audio samples (attacks and original)


###  1. Description

#### Det_DeepSpeech.py

We generate multiple Gaussian noises and add them to the result of acoustic model in the ASR system (called logits). 
Each logit sequence is transcribed to character-level transcription using beam-search decoding. After that, we calculate the Character Error Rate (CER) for original logits transcription and each noised logits transcription. Finally, the sum of each CER is over than threshold, we consider it an attack. Below figure is overall architecture we proposed.

![ex_screenshot](./figs/logit_noising_architecture.png)

#### logit_analysis_ACSAC.ipyb

For generating robust detection system, we select appropriate noise distribution. 
This file help us to calculate the 1-k logit-gap distribution and 1-k inversion probability.

### 2. Installation

Download the DeepSpeech v0.1.1 :
https://github.com/mozilla/DeepSpeech/tree/v0.1.1

Download the pre-trained model(v0.1.0) :
https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz

Also, DeepSpeech requires native client to execute ASR system. However, the
native client for v0.1.1 is expired, so you should rebuild it by yourself.
Install native client:
https://github.com/mozilla/DeepSpeech/tree/v0.1.1/native client#readme

### 3. Prerequisites

Detect_DeepSpeech file execution environment is as follows
* Python 3.6
* CUDA 8
* CUDNN 6
* 1 Titan V GPU
* pip3 install editdistance (*as same as DeepSpeech v0.1.1)

logit_analysis_ACSAC file execution environment is as follows below
* Python 3.6
* CUDA 9
* CUDNN 7.6
* 1 Titan V GPU
* pip3 install numpy scipy tensorflow-gpu==1.8.0 pandas python_speech_features, matplotlib


To execute our detection system, you have to install a native_client file[native_client]https://github.com/mozilla/DeepSpeech/blob/v0.1.1/native_client
However, the artifact expired, we should rebuid it. 

Enter the DeepSpeech original github site [DeepSpeech_v0.1.1]https://github.com/mozilla/DeepSpeech/tree/v0.1.1
You can install deepspeech by following the steps from the link above.

If you rebuild DeepSpeech v0.1.1 execute Detect_DeepSpeech.py
```
python3 python -u Detect_DeepSpeech.py  --checkpoint_dir "$checkpoint_dir" 
```
