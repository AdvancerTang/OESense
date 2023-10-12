# OESense
&nbsp;Since the current push-type and knock-type wireless Stereo control methods are insensitive and inconvenient, this project adopts gesture control to solve these problems.The control of in-ear headphones is based on  occlusion effect for in-ear human sensing,we hope taht in this way the TWS will be more beneficial to users.
## Class
  The classes are simplified as follows:

  _class1_ including[left forehead/left temple/left cheek/left jaw];  

  _class2_ including[right forehead/right temple/right cheek/right jaw];

  _class3_ including[nose/ philtrum/ chin];

  _class4_ including[middle forehead].

## Feature
The tested feature include time frequency, STFT spectrum(129), mel spectrogram(128), currently, mel is the best

## Data
Compared with two channel feature, singal channel of channel[0] is better, we use the data of channel[0] and channel[1] to enlarge the dataset.

Data name: S_[Person] _Ges_ [origin label] _ [new label] -[Channel]-[Noise type]-[pitch num]  exp:S10_Ges_1_1-1-c-9.wav

[paper](https://www.researchgate.net/publication/352713439_OESense_employing_occlusion_effect_for_in-ear_human_sensing)
