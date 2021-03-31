import numpy as np 
import matplotlib.pyplot as plt 
import random
from skimage.util import random_noise

def PlotSeis(data, num=0, save=False):

    size = np.array(data[0]).shape[1]

    fig, axs = plt.subplots(1, len(data), figsize=(len(data*4), 7))

    vmin = -np.max(data[0][num])
    vmax = np.max(data[0][num])
    # Looping over datasets to compare
    for j in range(len(data)):
        im = axs[j].imshow(data[j][num].reshape(size,size).T, aspect='auto', interpolation='nearest',
            vmin=vmin, vmax=vmax, cmap='gray')

    # fig.colorbar(axs[-1], im)
    if save:
        file_name = input("file name:")
        plt.savefig('./results/images/%s_start%s.png'%(file_name,start))
