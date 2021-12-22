'''
use the pytorch-fid to calculate the FID score.
calculate with 10k images, in different 9900 epochs, separately.
'''
from FID.calc_fid import *

# %%
PATH = "/home/xchen/GANs/WGAN-div-PyTorch-/samples/"
FILE_NAME = "1204_mnist_matsumoto_1"

fid_one_list(PATH, FILE_NAME)
