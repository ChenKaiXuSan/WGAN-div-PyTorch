# WGAN-div-PyTorch
Pytorch implementation of Wasserstein Divergence for GANs (WGAN-div)

## Overview
This repository contains an Pytorch implementation of WGAN-DIV.
With full coments and my code style.

## About WGAN-div
If you're new to Wasserstein Divergence for GANs (WGAN-div), here's an abstract straight from the paper[3]:


## Dataset 
- MNIST
`python3 main.py --dataset mnist --channels 1`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3`

## Implement
```python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Generator                                --                        --
├─Sequential: 1-1                        [64, 512, 4, 4]           --
│    └─ConvTranspose2d: 2-1              [64, 512, 4, 4]           819,200
│    └─BatchNorm2d: 2-2                  [64, 512, 4, 4]           1,024
│    └─ReLU: 2-3                         [64, 512, 4, 4]           --
├─Sequential: 1-2                        [64, 256, 8, 8]           --
│    └─ConvTranspose2d: 2-4              [64, 256, 8, 8]           2,097,152
│    └─BatchNorm2d: 2-5                  [64, 256, 8, 8]           512
│    └─ReLU: 2-6                         [64, 256, 8, 8]           --
├─Sequential: 1-3                        [64, 128, 16, 16]         --
│    └─ConvTranspose2d: 2-7              [64, 128, 16, 16]         524,288
│    └─BatchNorm2d: 2-8                  [64, 128, 16, 16]         256
│    └─ReLU: 2-9                         [64, 128, 16, 16]         --
├─Sequential: 1-4                        [64, 64, 32, 32]          --
│    └─ConvTranspose2d: 2-10             [64, 64, 32, 32]          131,072
│    └─BatchNorm2d: 2-11                 [64, 64, 32, 32]          128
│    └─ReLU: 2-12                        [64, 64, 32, 32]          --
├─Sequential: 1-5                        [64, 1, 64, 64]           --
│    └─ConvTranspose2d: 2-13             [64, 1, 64, 64]           1,024
│    └─Tanh: 2-14                        [64, 1, 64, 64]           --
==========================================================================================
Total params: 3,574,656
Trainable params: 3,574,656
Non-trainable params: 0
Total mult-adds (G): 26.88
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 127.93
Params size (MB): 14.30
Estimated Total Size (MB): 142.25
==========================================================================================

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Discriminator                            --                        --
├─Sequential: 1-1                        [64, 64, 32, 32]          --
│    └─Conv2d: 2-1                       [64, 64, 32, 32]          1,024
│    └─LeakyReLU: 2-2                    [64, 64, 32, 32]          --
├─Sequential: 1-2                        [64, 128, 16, 16]         --
│    └─Conv2d: 2-3                       [64, 128, 16, 16]         131,072
│    └─BatchNorm2d: 2-4                  [64, 128, 16, 16]         256
│    └─LeakyReLU: 2-5                    [64, 128, 16, 16]         --
├─Sequential: 1-3                        [64, 256, 8, 8]           --
│    └─Conv2d: 2-6                       [64, 256, 8, 8]           524,544
│    └─BatchNorm2d: 2-7                  [64, 256, 8, 8]           512
│    └─LeakyReLU: 2-8                    [64, 256, 8, 8]           --
├─Sequential: 1-4                        [64, 512, 4, 4]           --
│    └─Conv2d: 2-9                       [64, 512, 4, 4]           2,097,664
│    └─BatchNorm2d: 2-10                 [64, 512, 4, 4]           1,024
│    └─LeakyReLU: 2-11                   [64, 512, 4, 4]           --
├─Sequential: 1-5                        [64, 1, 1, 1]             --
│    └─Conv2d: 2-12                      [64, 1, 1, 1]             8,192
==========================================================================================
Total params: 2,764,288
Trainable params: 2,764,288
Non-trainable params: 0
Total mult-adds (G): 6.51
==========================================================================================
Input size (MB): 1.05
Forward/backward pass size (MB): 92.28
Params size (MB): 11.06
Estimated Total Size (MB): 104.38
==========================================================================================

```
## Usage
- MNSIT  
`python3 main.py --dataset mnist --channels 1 --version [version] --batch_size [] --adv_loss [] >logs/[log_path]`
- FashionMNIST  
`python3 main.py --dataset fashion --channels 1 --version [version] --batch_size [] --adv_loss [] >logs/[log_path]`
- Cifar10  
`python3 main.py --dataset cifar10 --channels 3 -version [version] --batch_size [] --adv_loss [] >logs/[log_path]`

## FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

For the FID, I use the pytorch implement of this repository. [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid)

For the 10k epochs training on different dataset, compare with about 10000 samples, I get the FID: 

| dataset | wgan-div |
| ---- | ---- |
| MNIST | 84.45772845527125(5900epoch) |
| FASHION-MNIST | 177.41489545734555(500epoch) | 
| CIFAR10 | 54.480231280904434(1000epoch) |
 
> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 

## Reference
1. [WGAN](https://arxiv.org/abs/1701.07875)
2. [WGAN-GP](https://arxiv.org/abs/1704.00028)
3. [WGAN-DIV](https://arxiv.org/abs/1712.01026)
4. [DCGAN](https://arxiv.org/abs/1511.06434)
