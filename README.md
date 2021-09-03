### Detecting_Audio_Adversarial_Examples_With_Logit_Noising

This repository contains code of ACSAC 2021 paper "Detecting Audio Adversarial Examples with Logit Noising".

In thi repository, we upload Det_DeepSpeech.py, logit_analysis_ACSAC.ipyb, and audio samples (attacks and original)

#### Det_DeepSpeech.py

In this file, we generate multiple Gaussian noises and add them to the result of acoustic model in the ASR system (called logits). 
Each logits sequence is transcribed to character-level transcription using beam-search decoding. After that, we calculate the Character Error Rate (CER) for original logits transcription and each noised logits transcription. Finally, the sum of each CER is over than threshold, we consider it an attack. Below figure is overall architecture we proposed.

Detect_DeepSpeech file execution environment is as follows below
* Python 3.6
* CUDA 8
* CUDNN 6
* 1 Titan V GPU
* pip3 install editdistance (*as same as DeepSpeech v0.1.1)

![ex_screenshot](./figs/logit_noising_architecture.png)


#### logit_analysis_ACSAC.ipyb

For generating robust detection system, we select appropriate noise distribution. 
This file help us to calculate the 1-k logit-gap distribution and 1-k inversion probability.

logit_analysis_ACSAC file execution environment is as follows below
* Python 3.6
* CUDA 9
* CUDNN 7.6
* 1 Titan V GPU
* pip3 install numpy scipy tensorflow-gpu==1.8.0 pandas python_speech_features, matplotlib

![ex_screenshot](./figs/distribution.png)

Finally, using 1-k inversion probability, we can caculate total inversion probability
![ex_screenshot](./figs/inversion_probability.png)

To execute our detection system, you have to install a native_client file[native_client]https://github.com/mozilla/DeepSpeech/blob/v0.1.1/native_client
However, the artifact expired, we should rebuid it. 

Enter the DeepSpeech original github site [DeepSpeech_v0.1.1]https://github.com/mozilla/DeepSpeech/tree/v0.1.1
You can install deepspeech by following the steps from the link above.

If you rebuild DeepSpeech v0.1.1 execute Detect_DeepSpeech.py
```
python3 python -u Detect_DeepSpeech.py  --checkpoint_dir "$checkpoint_dir" 
```
