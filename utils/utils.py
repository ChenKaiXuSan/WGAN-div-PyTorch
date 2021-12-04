# %% 
import os
import numpy as np 
import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision.utils import save_image

import shutil

# %%
def del_folder(path, version):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if os.path.exists(os.path.join(path, version)):
        shutil.rmtree(os.path.join(path, version))
    
def make_folder(path, version):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and set grad to false

    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.

    Returns:
        tensor: tensor in gpu and set grad to false 
    '''    
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x

def var2tensor(x):
    '''
    put date to cpu

    Args:
        x (tensor): input tensor 

    Returns:
        tensor: put data to cpu
    '''    
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def str2bool(v):
    return v.lower() in ('true')

def to_Tensor(x, *arg):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.LongTensor
    return Tensor(x, *arg)

# custom weights initialization called on netG and netD
def weights_init(m):
    '''
    custom weights initializaiont called on G and D, from the paper DCGAN

    Args:
        m (tensor): the network parameters
    '''    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample_one_image(sample_path, real_images, fake_images, epoch, number=0):

    make_folder(sample_path, str(epoch) + '/real_images')
    make_folder(sample_path, str(epoch) + '/fake_images')
    real_images_path = os.path.join(sample_path, str(epoch), 'real_images')
    fake_images_path = os.path.join(sample_path, str(epoch), 'fake_images')

    # saved image must more than 10000 sheet
    # the number of the generaed images must larger than 10000 for the FID score.
    while len(os.listdir(real_images_path)) <= 10000:
        for i in range(real_images.size(0)):
            # save real image
            one_real_image = real_images[i]
            save_image(
                one_real_image.data, 
                os.path.join(real_images_path, '{}_real.png'.format(number)),
                normalize=True
            )

            # save fake image
            one_fake_image = fake_images[i]
            save_image(
                one_fake_image.data,
                os.path.join(fake_images_path, '{}_fake.png'.format(number)),
                normalize=True
            )

            number += 1
        
        if number == 10000:
            break

    return number

def save_sample(path, images, epoch):
    '''
    save the tensor sample to nrow=10 image

    Args:
        path (str): saved path
        images (tensor): images want to save
        epoch (int): now epoch int, for the save image name
    '''    
    save_image(images.data[:100], os.path.join(path, '{}.png'.format(epoch)), normalize=True, nrow=10)

"""
* compute the wasserstein divergence for gradient penalty from the wgan-div
"""
def compute_gradient_penalty_div(real_out, fake_out, real_images, fake_images, k=2, p=6):
    '''
    compute the gradient with the paper wgan-div

    Args:
        real_out ([tensor]): D output from the real images, (*,)
        fake_out ([tensor]): D output from the fake images, (*,)
        real_images ([tensor]): real images from the dataset
        fake_images ([tensor]): fake images from the G(z)
        k (int, optional): [description]. Defaults to 2.
        p (int, optional): [description]. Defaults to 6.

    Returns:
        [tensor]: computed the gradient penalty
    '''    
    real_grad = autograd.grad(
        outputs=real_out,
        inputs=real_images,
        grad_outputs=torch.ones(real_images.size(0)).cuda(),
        retain_graph=True,
        create_graph=True,
        only_inputs=True          
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

    fake_grad = autograd.grad(
        outputs=fake_out,
        inputs=fake_images,
        grad_outputs=torch.ones(fake_images.size(0)).cuda(),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

    gradient_penalty = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
    return gradient_penalty