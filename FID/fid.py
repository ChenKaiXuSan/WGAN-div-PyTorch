'''
use the pytorch-fid to calculate the FID score.
calculate with 10k images, in different 9900 epochs, separately.
'''
from calc_fid import *

# %%
PATH = "/GANs/WGAN-div-PyTorch-/samples/"
FILE_NAME = "1204_mnist_matsumoto_1"

fid_one_list(PATH, FILE_NAME)

# fid_all_list("/GANs/WGAN-div-PyTorch-/samples/1216_cifar10_bs64_matsumoto_1")
