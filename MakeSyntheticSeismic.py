import numpy as np 
import matplotlib.pyplot as plt 
from pylops.utils.seismicevents import makeaxis, linear2d
from pylops.utils.wavelets import ricker
import random

def MakeSeismic(samples, noise_level=0.3 ):

    """Simple generation of noisy synthetic linear seismic events. 
        Input:
        samples =  Number of samples in your dataset you want
        noise_level = Guassian noise level that gets added to seismic
        
        Output: 
        clean_signal, noise, noisy_signal"""


    # empty list to be filled with numpy arrays
    clean_signal = []
    noise = []
    noisy_signal = []

    # Parameters for the seismic canvas
    par = {'ox':0, 'dx':12.5, 'nx':128, # offsets
       'ot':0, 'dt':0.004, 'nt':128, # time
       'f0': random.randint(5,30), 'nfmax': 50}
    
    # Make canvas
    t, t2, x, y = makeaxis(par)
    # Make wavelet
    wav = ricker(np.arange(41) * par['dt'],f0=par['f0'])[0]

    # Parameters for events
    v = 1500
    ang_range = 70
    amp_range = 2
    i = 0
    while i <= samples: 
        # Time of events
        t0 = [random.uniform(t.min(),t.max())*0.8, random.uniform(t.min(),t.max())*0.8, random.uniform(t.min(),t.max())*0.8]
        # Angle of events
        theta = [random.uniform(-ang_range,ang_range),random.uniform(-ang_range,ang_range),random.uniform(-ang_range,ang_range)]
        # Ampltidues of events
        amp = [random.uniform(-amp_range,amp_range), random.uniform(-amp_range,amp_range), random.uniform(-amp_range,amp_range)]
        # Making events
        mlin, mlinwav = linear2d(x, t, v, t0,theta, amp, wav)
        # Creating noise
        n = np.random.normal(loc=0,scale=1.0,size=mlinwav.shape)*random.uniform(0.1,0.7)
        s = mlinwav
        # Adding noise
        ns = s+n

        clean_signal.append(s)
        noise.append(n)
        noisy_signal.append(ns)
        i +=1

    return np.array(clean_signal), np.array(noise), np.array(noisy_signal)

def PlotSeis(data, num=0, save=False):

        # Parameters for the seismic canvas
    par = {'ox':0, 'dx':12.5, 'nx':128, # offsets
       'ot':0, 'dt':0.004, 'nt':128, # time
       'f0': random.randint(5,30), 'nfmax': 50}
    
    # Make canvas
    t, t2, x, y = makeaxis(par)

    fig, axs = plt.subplots(1, len(data), figsize=(12, 5))
    # Looping over datasets to compare
    for j in range(len(data)):
        axs[j].imshow(data[j][num].T, aspect='auto', interpolation='nearest',
            vmin=-2, vmax=2, cmap='gray',
            extent=(x.min(), x.max(), t.max(), t.min()))

    if save:
        file_name = input("file name:")
        plt.savefig('./results/images/%s_start%s.png'%(file_name,start))
