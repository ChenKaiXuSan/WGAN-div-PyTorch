# %%
PATH = "/home/xchen/GANs/WGAN-div-PyTorch-/samples/1204_mnist_matsumoto_1/"
fid='python3 -m pytorch_fid'

# %%
import subprocess
import shlex

dict = {}

for i in range(0, 10000, 100):

    real_path = ' ' + PATH + str(i) + '/real_images'
    fake_path = ' ' + PATH + str(i) + '/fake_images'

    command_line = fid + real_path + fake_path

    args = shlex.split(command_line)

    res = subprocess.run(args, shell=False, stdout=subprocess.PIPE, text=True)
    
    dict[i] = res.stdout

# %%
import json 
import pprint

with open('reslut.log', "w") as tf:
    # tf.write(json.dumps(dict, sort_keys=True, indent=4))

    print('\n')

    pprint.pprint(sorted(dict.items(), key=lambda kv:kv[1]), stream=tf)
    # print(sorted(dict.items(), key=lambda kv:kv[1]), file=tf)



