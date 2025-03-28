---
title: '跑代码记录'
publishDate: 2025-03-28
description: 'SurgicalGaussian - Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction'
tags:
 - essay
 - Medical Imaging
language: 'Chinese'
heroImage: { src: './论文.png', color: '#E3CAB5' }
---

## 指路
这次跑的实验是[这个](https://github.com/xwx0924/SurgicalGaussian?tab=readme-ov-file),这篇论文的笔记在[这](https://laurie-hxf.xyz/blog/论文笔记)

## 前言


不得不吐槽一下跑实验原来这么麻烦，我大概在周日的时候就开始跑，一直配环境到今天，每一天的命令行都是满屏红色，看着都要红温。多亏了师兄的帮助，帮我这个菜鸡解决好多问题。

当环境一直配不好的时候，我就纳闷了，为什么这些研究人员不可以把他们的这些环境封装成一个docker，这样就不用再让别人折腾配环境。后来问师兄，原来就是懒，他们开源这些代码就只是证明他们可以跑出来，并不是要商业化，把配环境的麻烦交给别人。我真服了。

下面就讲一下我配这个环境的走过的路😭

## 配置

- Ubuntu 18.04.6 LTS (GNU/Linux 5.4.0-150-generic x86_64)
- pytorch 1.12.0
- cuda 11.6

在配置之前安装conda是必要的，避免这些环境污染别的，然后就是mamba！！！这个巨重要，极大加速安装环境的速度，而且他能更快地找到合适的包版本。这个真的极大优化你的体验，之前一直用conda，安装包巨慢，关键是他还很容易报错，安了这么久，结果给我报一堆红色的错误，当时真的破防。

## 安装

然后就开始跟着github少的可怜的指示来安装

```shell
git clone https://github.com/xwx0924/SurgicalGaussian.git
cd SurgicalGaussian

conda create -n SurgicalGaussian python=3.7 
conda activate SurgicalGaussian

# install pytorch and others.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
# You also need to install the pytorch3d library to compute Gaussian neighborhoods.

# You can follow 4DGS to download depth-diff-gaussian-rasterization and simple-knn.
pip install -e submodules/depth-diff-gaussian-rasterization  
pip install -e submodules/simple-knn
```

#### Pytorch

这里我安装的是pytorch 1.12.0+cu11.6 可以上[pytorch官网](https://pytorch.org/get-started/previous-versions/)来找对应安装版本的指令,我这里用的指令就是
```shell
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

这里用了个python程序来检验安装的版本是不是想要安装的版本

```python
import torch
print(torch.__version__)          # PyTorch 版本
print(torch.version.cuda)         # PyTorch 编译时使用的 CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
```

输出就是

```shell
1.12.0
11.6
True
```


#### Pytorch3D

然后就根据指引安装requirements.txt，安完之后就到了另一个坑，指引只是轻描淡写的写了一下要安装pytorch3d，但这个库可一点都不好安装。

指引没有提怎么能安好这个库，这种情况下就去搜索这个库的[github仓库](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)，里面提到要先安装一些环境，再安装pytorch3d这个库，先安装iopath，然后如果你的cuda版本低于11.7，你就要安装另外一个库。这里也建议用manba安装
![alt text](./截屏2025-03-28%2021.45.40.png)

```shell
mamba install -c iopath iopath
mamba install -c bottler nvidiacub
# Anaconda Cloud
mamba install pytorch3d -c pytorch3d
```

反正遇到conda的指令都替换为mamba，真的太折磨了。

#### Submodels

安装好后开始安装这两个submodel，注意的是，这两个model并没有包含在原仓库的代码里你必须跑到对应的别的库里面安装这两个model，然后这里又一个坑，指引说他借鉴了3D和4D两个仓库，这两个仓库里面都有这两个submodel，关键是这两个库里面的model版本是不一样的。原仓库安装的其实是[4D](https://github.com/hustvl/4DGaussians)的submodel，这里就将这个仓库clone然后把submodel拷贝到原仓库中

这下终于可以安装这两个model，结果又报错，他可能报错你的pytorch版本和cuda版本不匹配，应为我安装的pytorch版本是基于cuda11.6的，但我的系统用的是10.1

可以用

```shell
nvcc --version
```

来看当前使用的cuda版本到底是什么，然后其实才发现原来cuda版本都没有安装，然后跑到[nvidia官网](https://developer.nvidia.com/cuda-11-6-0-download-archive)安装
根据提示来选择，最后一个选择local
```shell
wget [https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run](https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run)

sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit --toolkitpath=~/cuda-11.6 --defaultroot=~/cuda-11.6
```

好像这里还要export一下，具体指令忘记了。这里值得注意的是原本我只在conda里面安装了

```shell
mamba install nvidia/label/cuda-11.6.1::cuda-toolkit
```

然后后面就会报错缺少什么.h文件，于是就替换成了这个命令。
再用这个指令检查一下
```shell
nvcc --version
```

然后后面就开始安装submodels，这里应该就没有问题。

#### Train

安装完成以后以为万事大吉，环境终于配好了，结果还有报错。但我满心欢喜的想要运行训练的指令，结果又报错。说是缺mmcv这个库，安装库真的很痛苦，这里又踩坑。看[官网](https://mmcv.readthedocs.io/zh-cn/2.x/get_started/installation.html),然后注意的是安装的版本要是1.7，不要安装2.0之后的版本，可能原仓库用的就是之前的版本。推荐使用官方推荐的mim安装方式

```shell
pip install -U openmim
mim install mmcv-full==1.7.0
```

安装好这个库后，我又运行了训练的命令，结果又有库没有安装，tinycudann，这玩意真是个毒瘤，网上一堆人安装这个库报错，我安装了好久，结果等我安装好后又报错说什么`Could not find compatible tinycudann extension for compute capability 75.`  md，这里的解决办法就是找到报错的那个程序，把这个库给注释掉，就是这么无语，这个库根本没有用到。

到此应该就没有问题了，还要注意的是，数据集原仓库也没有给，要自己去给的链接那里下载，然后安装的文件排列要遵循指引中提到的，

然后应该就可以~~顺利~~的运行训练指令，看到进度条开始跑的那一瞬间都快哭了，我真服了。跑完之后跑render指令，这里又要注意的是render里面还有一个库要安装，一看又是没有用上的，就注释掉了。然后就把剩下的指令跑完了就行了，跑出个结果。

![alt text](./截屏2025-03-28%2023.55.54.png)
## 感悟

1. 经过这个贼麻烦的配置环境，还有要提的，就是一旦遇到报错了，先看一下报错信息是什么，看自己能不能解决。解决不了，把报错信息google一下，然后就会发现，原来已经有这么多人踩过坑，一般都在github issue里面，一般都已经有了解决方法。下策就是直接把报错喂给ai，这种做法真的很痛苦，如果你按照ai的指令瞎跑一堆，跑到最后，你的问题没有解决，但是你的环境已经混乱了。

2. 还有就是当你连接服务器跑代码的时候最好用tmux，这个可以直接在后台跑，不用怕跑着跑着电脑和服务器断开连接，结果你再也不能和那个进程连接上。



