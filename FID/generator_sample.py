#%%
import sys
sys.path.append("..")

#%%
import torch.nn as nn
import torch 
from utils.utils import *

from models.wgan_div import Generator
# %%
Generator = Generator(channels=3).cuda()

fixed_z = tensor2var(torch.randn(1000, 100)) # （*, 100）

# Generator.eval()
# fake_images = Generator(fixed_z)

# %%
checkpoint = torch.load("/GANs/WGAN-div-PyTorch-/checkpoint/1216_cifar10_bs64_matsumoto_1/10000.pth.tar")
Generator.load_state_dict(checkpoint['G_state_dict'])
# %%
fake_images = Generator(fixed_z)
# %%
