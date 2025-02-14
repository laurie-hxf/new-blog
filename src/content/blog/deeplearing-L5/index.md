
---
title: 'Neural Networks'
publishDate: 2025-02-014
description: 'UMich EECS 498-007 Deep learning-Neural Networks'
tags:
  - Neural Networks
  - DeepLearing
language: 'Chinese'
heroImage: { src: './截屏2025-02-09 16.41.24.png', color: '#64574D' }
---

## Problem: Linear Classifiers aren’t that powerful

### #1 Feature Transforms
我们之前讲过linear classifier是线性的，所以他不能识别一些非线性的图案。但是我们可以通过一些方法将我们要识别的数据转化成线性的，这样我们就可以利用Linear Classifiers来识别
![alt text](./截屏2025-02-09%2015.51.34.png)

### #2 Color Histogram
我们将图片转化为颜色直方图，这样可以忽略物体在照片中的空间位置，根据颜色来识别物体，把图片中的颜色转化为向量然后训练。
![alt text](./截屏2025-02-09%2015.57.34.png)

### #3 Histogram of Oriented Gradients (HoG)

HoG的核心思想是**通过捕捉图像中物体轮廓和边缘的形状信息**来提取特征。图像的梯度方向和幅度可以反映出物体的边缘、纹理等结构信息，这些信息对物体的识别和分类非常重要。

![alt text](./截屏2025-02-09%2015.59.11.png)

### #4 Bag of Words (Data-Driven!)

我们冲数据集中每个图片提取一些块，然后这些块组成一个 codebook，然后我们就可以将图片表示为这个图片有多少个codebook中这一个块的个数，以此类推
![alt text](./截屏2025-02-09%2016.13.30.png)

---
## Neural Networks

上面说的这些都是图片的某些特征，我们不只用单独的特征来识别图片，我们用多个特征组合起来形成一个长特征向量来表示一张图片
![alt text](./截屏2025-02-09%2016.17.44.png)
以前的想法就是将一个系统分为两部分，一部分就是特征的提取，一部分就是训练部分
神经网络的动机就是最大化提高图像分类的能力，最大的区别就是他用一整个系统共同来调整这两部分

现在就看一下神经网络的简单例子

$$
Linear\ Classifiers:f = Wx+b
$$

$$
2\ layer\ Neural\ Networks:f = W_2max(0,W_1x+b_1)+b_2
$$

### Fully-connected neural network

由于x中的每个元素都会对h中的每个元素造成影响，h中的也会对s造成影响，神经网络的每一层都是相互连接的，所以将这种神经网络称为Fully-connected neural network，也叫多层感知机(MLP)
![alt text](./截屏2025-02-09%2016.41.24.png)

max那部分被称为**激活函数**，如果我们没有那部分，我们的函数变为$s=W_2W_1x$ 这时他仍然是一个Linear Classifiers，所以我们要在两个矩阵之间加一个非线性的函数。当然这种激活函数可以有很多种，不只是max这种，但max是用的最广泛的激活函数。

激活函数最重要的作用就是将分类可以变的不再线性，
![alt text](./截屏2025-02-09%2022.39.21.png)