# tensorflow performance gpu vs cpu
As I have not used dedicated graphic card previously, I have some doubts about the performance boost introduced by tensorflow gpu.

refer to [this link](https://medium.com/@andriylazorenko/tensorflow-performance-test-cpu-vs-gpu-79fcd39170c) for a more detailed comparison.

A simple test uses my PC and macbook, and summary is listed below. In general, tensorflow gpu performs 15x than cpu and desktop cpu >> laptop cpu.

|device                                | examples per second|
| -------------                        |:-------------:|
|intel i7-3770 + 16GB PC               | 600           |
|nvidia gtx 1060 6GB                   | 8000          |
|Intel Core i7 macbook 2.5GHz          | 20            |

## tensorflow gpu
7900 examples per second
```
git clone https://github.com/tensorflow/models.git
cd models/tutorials/image/cifar10
python cifar10_train.py
>> Downloading cifar-10-binary.tar.gz 100.0%
Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2018-09-09 23:09:00.743632: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2018-09-09 23:09:01.065173: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-09-09 23:09:01.065646: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.835
pciBusID: 0000:01:00.0
totalMemory: 5.91GiB freeMemory: 5.84GiB
2018-09-09 23:09:01.065664: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-09 23:09:01.292559: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-09 23:09:01.292596: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0
2018-09-09 23:09:01.292604: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N
2018-09-09 23:09:01.292785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5620 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
2018-09-09 23:09:06.600383: step 0, loss = 4.67 (213.6 examples/sec; 0.599 sec/batch)
2018-09-09 23:09:06.868952: step 10, loss = 4.63 (4765.9 examples/sec; 0.027 sec/batch)
2018-09-09 23:09:07.032658: step 20, loss = 4.46 (7818.8 examples/sec; 0.016 sec/batch)
2018-09-09 23:09:07.194632: step 30, loss = 4.60 (7902.7 examples/sec; 0.016 sec/batch)
2018-09-09 23:09:07.355291: step 40, loss = 4.37 (7967.4 examples/sec; 0.016 sec/batch)
```

## tensorflow + desktop cpu

600 examples/second

create a new testing env and install tensorflow cpu only

```
conda create --name machinelearning_test_cpu_env python=3

source activate machinelearning_test_cpu_env

conda install -c conda-forge tensorflow 

cd models/tutorials/image/cifar10
python cifar10_train.py
(machinelearning_test_cpu_env) dadoudou@dadoudou-ThinkCentre:~/machinelearning/tensorflow/models/tutorials/image/cifar10$ python cifar10_train.py
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2018-09-09 23:16:09.784249: step 0, loss = 4.67 (628.2 examples/sec; 0.204 sec/batch)
2018-09-09 23:16:12.212964: step 10, loss = 4.61 (527.0 examples/sec; 0.243 sec/batch)
2018-09-09 23:16:14.542684: step 20, loss = 4.51 (549.4 examples/sec; 0.233 sec/batch)
2018-09-09 23:16:16.878033: step 30, loss = 4.45 (548.1 examples/sec; 0.234 sec/batch)
```

## tensorflow + macbook cpu
average 20 examples/sec
```
# run sample training code
git clone https://github.com/tensorflow/models.git
cd models/tutorials/image/cifar10
python cifar10_train.py

# result
>> Downloading cifar-10-binary.tar.gz 100.0%
Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2018-09-17 22:53:56.098168: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that thisTensorFlow    binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-09-17 22:54:07.991817: step 0, loss = 4.68 (106.8 examples/sec; 1.198 sec/batch)
2018-09-17 22:55:14.520668: step 10, loss = 4.63 (19.2 examples/sec; 6.653 sec/batch)
2018-09-17 22:56:21.955433: step 20, loss = 4.51 (19.0 examples/sec; 6.743 sec/batch)
2018-09-17 22:57:29.898635: step 30, loss = 4.46 (18.8 examples/sec; 6.794 sec/batch)
2018-09-17 22:58:37.495308: step 40, loss = 4.38 (18.9 examples/sec; 6.760 sec/batch)
2018-09-17 22:59:45.772787: step 50, loss = 4.36 (18.7 examples/sec; 6.828 sec/batch)
2018-09-17 23:00:53.491838: step 60, loss = 4.30 (18.9 examples/sec; 6.772 sec/batch)
2018-09-17 23:02:02.508488: step 70, loss = 4.28 (18.5 examples/sec; 6.902 sec/batch)
2018-09-17 23:03:10.701406: step 80, loss = 4.21 (18.8 examples/sec; 6.819 sec/batch)
```