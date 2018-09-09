# learn machine learning
This is my note to learn machine learning.

General background, took some machine learning courses in the graduate school, understand general machine learning concepts (artificial neural network, genetic algorithm, bayesian learning, natural language processing, etc.) and have applied numeric convex analysis (mainly first-order and second-order gradient descent) and hyper parameter optimization in the ph.d research.

Recently, I would like to take on the latest advancement in deep learning (neural network) and big data, and see if these two can lead to some proof-of-concept prototype in automatic speech recognition (ASR).

# development environment for deep learning
Even as a software engineer, it still took me some time to make hardware and software work together properly. So it is necessary to document these steps and not to repeat myself again in the future. It may be helpful to others as well.

A new graphic card is added to my old PC and this is the set up that is working for me. As of 09/08/2018, Nvidia GPU driver is installed without login loop. python3 is installed in virtual environment. Tensorflow can pick up the gpu just fine and can use up almost all 6GB when performing training task.

```
cpu: intel core i7-3770
memory: 16GB
hard disk: 1TB
graphic card: evga geforce gtx 1060 6GB
operating system: ubuntu 18.04
nvidia driver: nvidia-396
cuda library: 9.2
cudnn: v7.2
python: 3.6
anaconda: version 5.2.0 for python 3
tensorflow-gpu: 1.10
```

## graphic card choice
the most important is to choose a NVIDIA GPU graphic card that supports compute capability 3.0 or higher. refer to this link to choose a gpu that satisfies the [nvidia compute capability requirement](https://developer.nvidia.com/cuda-gpus).

According to this article about [how to choose a gpu for deep learning](https://blog.slavv.com/picking-a-gpu-for-deep-learning-3d4795c273b9). Conclusion: GeForce GTX 1050 is the entry-level card, and gtx 1060 will get you started in deep learning. gtx 1070/1080 or double is for kaggle competition.

EVGA GeForce GTX 1060 is my budget choice, because:
* it supports compute capability 6.1, so that tensorflow can be installed
* has 6GB memory and the rule of thumb is to let memory vs gpu memory = 2 : 1
* there is only one PCI-E slot in my old PC and the old PC has standard form (not the small form one), so no need to install multiple gpu

## install ubuntu 18.04
straight-forward

## install proprietary nvidia driver in ubuntu 18.04
to be continued

# automatic speech recognition (ASR)
Hidden markov model (HMM) was traditionally used in ASR and achieved wide acceptance. I would like to see if the recent advancement in deep learning can do a better job in ASR area, and perform some comparison with the traditional HMM approach.