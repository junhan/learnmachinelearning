# development environment for deep learning
Even as a software engineer, it still took me some time to make hardware and software work together properly. So it is necessary to document these steps and not to repeat myself again in the future. It may be helpful to others as well.

A new graphic card is added to my old PC and this is the set up that is working for me. As of 09/08/2018, Nvidia GPU driver is installed without login loop or black screen. python3 is installed in virtual environment via conda. Tensorflow can pick up the gpu just fine and can use up almost all 6GB when performing training task.

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
* it has 6GB memory and the rule of thumb is to let memory vs gpu memory = 2 : 1
* there is only one PCI-E slot in my old PC and the old PC has standard form (not the small form one), so it cannot install multiple gpus

# deep learning environment in ubuntu 18.04
## install ubuntu 18.04
straight-forward

## install cuda
1. install proprietary nvidia driver in ubuntu 18.04
    add graphic card driver repo and install nvidia-396

    ```
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt install nvidia-396
    ```

    use ` nvidia-smi` check driver installation result

    ```
    +------------------------------------------------------------    -----------------+
    | NVIDIA-SMI 396.44                 Driver Version: 396.44                       |
    |-------------------------------+---------------------- +----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A |    Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage |    GPU-Util  Compute M. |
    |===============================+====================== +======================|
    |   0  GeForce GTX 106...  Off  | 00000000:01:00.0 Off |                     N/A |
    |  0%   32C    P5    15W / 120W |      0MiB /  6055MiB |         2%      Default |
    +-------------------------------+---------------------- +----------------------+

    +-------------------------------------------------------------  ----------------+
    | Processes:                                                          GPU Memory |
    |  GPU       PID   Type   Process name                                Usage      |
    |   =============================================================  ================|
    |  No running processes found                                                    |
    +-------------------------------------------------------------  ----------------+
    ```

2. install cuda

   refer to [official document](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/tensorflow/)

   Navigate to [cuda download](https://developer.nvidia.com/cuda-downloads), download 

   ```
   # select linux, ubuntu, 17.10, deb (local)
   # file cuda-repo-ubuntu1710_9.2.148-1_amd64.deb is downloaded
   sudo dpkg -i cuda-repo-ubuntu1710_9.2.148-1_amd64.deb
   # it will prompt to install a public key, just follow the instructions
   sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub
   sudo apt update
   sudo apt install cuda
   # this will install the latest version of cuda, (cuda-9-2 at this moment)
   ```
   
   note, I have tried to install cuda from ubuntu repository with `sudo apt remove nvidia-cuda-toolkit`. It installs successfully, but tensorflow fails to detect the gpu. It installs cuda library under `/usr/lib/cuda`, which may cause problem for other cuda libraries.
   
   Therefore, follow the steps from official website, and installs the library in `/usr/local/cuda` and `/usr/local/cuda-9.2`. This will allow multiple cuda libraries.

   Reboot the PC to make sure the driver is installed properly and will not cause login loops or black screen.
3. update `~/.bashrc` and add cuda binary executables
   ```
   # cuda
   export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```
4. install cuDNN

   cuDNN is required by tensorflow, so install it as well.
   ```
   go to https://developer.nvidia.com/rdp/cudnn-download
   Download cuDNN v7.2.1 (August 7, 2018), for CUDA 9.2
   $ tar -xzvf cudnn-9.2-linux-x64-v7.2.1.38.tgz
   $ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
   $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
   $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
   ```
5. Prepare TensorFlow dependencies and required packages.
   ```
   $ sudo apt install libcupti-dev
   ```

## install tensorflow-gpu
1. install anaconda
   follow [official document](https://conda.io/docs/user-guide/install/linux.html)

   install anacoda for python 3 version 5.2.0
   it will install python 3 first and then install conda, and will not affect current ubuntu's python installation

   choose anaconda over pip is because it provides package dependency and version management. Choosing `anaconda` over `miniconda` is because `anaconda` includes all the packages by default and do not need to worry about dependencies.
2. create a conda environment
   ```
   # create env
   conda create --name machinelearning_env python=3
   # activate env
   source activate machinelearning_env
   ```
3. install tensorflow-gpu
   ```
   conda install -c anaconda tensorflow-gpu 
   ```
   it will install tensorflow-gpu, tensorflow-base and its dependency libraries.
4. Test tensorflow installation and check tensorflow can pick up the nvidia graphic card
   ```
   # activate the env
   source activate machinelearning_env
   # run python
   python
   # run this script
   [GCC 7.2.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> from tensorflow.python.client import device_lib
   >>> device_lib.list_local_devices()
   2018-09-09 22:27:46.923347: I    tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU    supports instructions that this TensorFlow binary was not    compiled to use: SSE4.1 SSE4.2 AVX
   2018-09-09 22:27:47.429684: I    tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897]    successful NUMA node read from SysFS had negative value (-1),    but there must be at least one NUMA node, so returning NUMA    node zero
   2018-09-09 22:27:47.430124: I    tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found    device 0 with properties:
   name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate   (GHz): 1.835
   pciBusID: 0000:01:00.0
   totalMemory: 5.91GiB freeMemory: 5.84GiB
   2018-09-09 22:27:47.430142: I    tensorflow/core/common_runtime/gpu/gpu_device.cc:1484]    Adding visible gpu devices: 0
   2018-09-09 22:27:51.903439: I    tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device    interconnect StreamExecutor with strength 1 edge matrix:
   2018-09-09 22:27:51.903475: I    tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0
   2018-09-09 22:27:51.903483: I    tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N
   2018-09-09 22:27:51.907456: I    tensorflow/core/common_runtime/gpu/gpu_device.cc:1097]    Created TensorFlow device (/device:GPU:0 with 5620 MB memory)    -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci    bus id: 0000:01:00.0, compute capability: 6.1)
   [name: "/device:CPU:0"
   device_type: "CPU"
   memory_limit: 268435456
   locality {
   }
   incarnation: 11996855961648024970
   , name: "/device:GPU:0"
   device_type: "GPU"
   memory_limit: 5893849088
   locality {
     bus_id: 1
     links {
     }
   }
   incarnation: 4248541454924357196
   physical_device_desc: "device: 0, name: GeForce GTX 1060 6GB,    pci bus id: 0000:01:00.0, compute capability: 6.1"
   ]

   # deactivate the env
   source deactivate
   ```

## install pytorch-gpu in ubuntu 18.04
As pytorch recently release their 1.0 version, install this framework as well.

Based on previous installation of cuda library, install pytorch via conda.
```
conda create --name python37 python=3.7
source activate python37
conda install -c pytorch pytorch
# when pytorch is installed, it will detect gpu automatically and install pytorch-gpu as its dependency
```

validate pytorch gpu usage
```
$ python
>>> import torch
>>> torch.cuda.is_available()
True
>>> x = torch.cuda.FloatTensor([1.0, 2.0])
>>> x
tensor([1., 2.], device='cuda:0')
```

[tensorflow vs pytorch](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)

# deep learning environment in macbook pro
for the sake of completeness, the steps for mac os is also listed here.

```
Model Name:	MacBook Pro
Processor Name:	Intel Core i7
Processor Speed:	2.5 GHz
Number of Processors:	1
Total Number of Cores:	2
Memory:	16 GB
Operating System: macOS high sierra
```

as there is no dedicated graphic card in macbook, install tensorflow cpu only and pytorch cpu only
1. install conda
   ```
   download conda v5.2.0 for python 3.7 https://repo.anaconda.com/archive/Anaconda3-5.2.0-MacOSX-x86_64.pkg
   install it following http://docs.anaconda.com/anaconda/install/mac-os/#macos-graphical-install
   ```
2. install tensorflow cpu   
   ```
   # create env
   conda create --name machinelearning_env python=3
   # activate the new env
   source activate machinelearning_env
   # install tensorflow cpu
   conda install -c anaconda tensorflow
   ```
3. validate tensorflow installation
   ```
   $ python
   Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:07:29)
   [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   >>> from tensorflow.python.client import device_lib
   >>> device_lib.list_local_devices()
   2018-09-17 22:47:16.264477: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow    binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
   [name: "/device:CPU:0"
   device_type: "CPU"
   memory_limit: 268435456
   locality {
   }
   incarnation: 13711008526164180218
   ]
   ```
4. install pytorch
   ```
   # create python3.7 environment
   # specify python=3 will keep the latest python3
   # conda is installed for python3 in macos, so the default python3 version is v3.7.0
   conda create --name python37  python=3
   conda activate python37
   conda install -c pytorch pytorch
   ```
   the stable version for pytorch: 0.4.1-py37_cuda0.0_cudnn0.0_1 pytorch
5. validate pytorch installation
   ```
   >>> import torch
   >>> a = torch.Tensor([[1,2],[3,4]])
   >>> print(a)
   ```

# machine learning project development
## install gradle and pygradle
use gradle to organize the whole project and pygradle is used for python projects
- gradle 4.10.2
- pygradle 0.8.10

refer to [this link](https://cs230-stanford.github.io/pytorch-getting-started.html) for python project layouts