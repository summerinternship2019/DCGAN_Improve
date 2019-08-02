import numpy as np
import matplotlib.pyplot as plt

samples = np.load('samples.npy')
samples[samples<=0] = 0
epc, bs, row, col, ch = samples.shape
print(epc,bs,row,col,ch)
#rows, cols = 24, 12
for ims in range(0,epc,100):
    fig, axs = plt.subplots(8, 4, figsize=(row*4/50,col*8/50))
    plt.subplots_adjust(hspace = .1, wspace = .1)
    cnt = 0
    for i in range(8):
        for j in range(4):
            axs[i,j].imshow(samples[ims, cnt,:,:,:],aspect='equal')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("imagesSel/pok_%d.png" % ims, bbox_inches='tight')
    plt.close()
