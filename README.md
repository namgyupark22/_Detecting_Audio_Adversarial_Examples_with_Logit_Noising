### Detecting_Audio_Adversarial_Examples_With_Logit_Noising

This repository contains code of ACSAC 2021 paper "Detecting Audio Adversarial Examples with Logit Noising".

In thi repository, we upload Det_DeepSpeech.py, logit_analysis_ACSAC.ipyb, and audio samples (attacks and original)

#### Det_DeepSpeech.py

We generate multiple gaussian noises and detect aduio adversarial examples in do_sing_file_inference function
![ex_screenshot](./figs/logit_noising_architecture.png)

#### logit_analysis_ACSAC.ipyb

We select noise & the number of noised instances to detect audio adversarial example. This file help user to calculate the 1-k logit-gap distribution and 1-k inversion probability.


![ex_screenshot](./figs/distribution.png)

Finally, using 1-k inversion probability, we can caculate total inversion probability
![ex_screenshot](./figs/inversion_probability.png)



If you already rebuild DeepSpeech v0.1.1 execute det_DeepSpeech.py

```
python3 python -u det_DeepSpeech.py --train_files "$any_file" --dev_files "$any_file" --test_files "$any_file" \
  --train_batch_size 1 \ --dev_batch_size 1 \ --test_batch_size 1 \  --n_hidden 2048 \ --epoch 1 \ --checkpoint_dir "$checkpoint_dir" \
```

If you have to build DeepSpeech v0.1.1, follow this website process[DeepSpeech_v0.1.1]https://github.com/mozilla/DeepSpeech/tree/v0.1.1


