import numpy as np 
import matplotlib.pyplot as plt 
from pylops.utils.seismicevents import makeaxis, linear2d
from pylops.utils.wavelets import ricker
import random

def MakeSeismic(samples,img_size=128,freq_low=5,freq_high=30,num_events=6):

    """Simple generation of noisy synthetic linear seismic events. 
        Input:
        samples =  Number of samples in your dataset you want
        
        Output: 
        clean_signal, noise, noisy_signal"""
    random.seed(101)
    # empty list to be filled with numpy arrays
    clean_signal = []
    noise = []
    noisy_signal = []
    # Parameters for the seismic canvas
    par = {'ox':0, 'dx':12.5, 'nx':img_size, # offsets
       'ot':0, 'dt':0.004, 'nt':img_size, # time
       'f0': random.randint(5,30), 'nfmax': 50}
    # Make canvas
    t, t2, x, y = makeaxis(par)
    # Make wavelet
    wav = ricker(np.arange(41) * par['dt'],f0=par['f0'])[0]
    # Parameters for events
    v = 1500
    ang_range = 50
    amp_range = 2
    i = 0
    amp_lim = 0.2
    while i < samples: 
        iEv = 0
        t0 = []
        theta = []
        amp = []
        while iEv <= num_events:
            # Time of events
            t0.append(random.uniform(t.min(),t.max())*0.7) 
            # Angle of events
            theta.append(random.uniform(-ang_range,ang_range))
            # Amplitude of events
            amp.append(random.uniform(-amp_range,amp_range))
            # clipping events to be above -0.2 and 0.2 
            if amp[iEv]<0:
                amp[iEv] = np.min([-amp_lim,amp[iEv]])
            else: 
                amp[iEv] = np.max([amp_lim,amp[iEv]])
            iEv+=1
        
        # Making events
        mlin, mlinwav = linear2d(x, t, v, t0,theta, amp, wav)
        # Creating noise
        n = np.random.normal(loc=0,scale=0.25,size=(img_size,img_size))

        # Adding noise
        s = mlinwav
        ns = s+n
        clean_signal.append(s)
        noise.append(n)
        noisy_signal.append(ns)
        i +=1

    return  np.array(clean_signal).reshape(samples,img_size,img_size,1),np.array(noise).reshape(samples,img_size,img_size,1),np.array(noisy_signal).reshape(samples,img_size,img_size,1)

def MakeSeismic_paper(samples,img_size=128,freq_low=5,freq_high=30,num_events=6):

    """Simple generation of noisy synthetic linear seismic events. 
        Input:
        samples =  Number of samples in your dataset you want
        
        Output: 
        clean_signal, noise, noisy_signal"""
    random.seed(101)
    # empty list to be filled with numpy arrays
    clean_signal = []
    noise = []
    noisy_signal = []
    # Parameters for the seismic canvas
    par = {'ox':0, 'dx':12.5, 'nx':img_size, # offsets
       'ot':0, 'dt':0.004, 'nt':img_size, # time
       'f0': 20, 'nfmax': 50}
    # Make canvas
    t, t2, x, y = makeaxis(par)
    # Make wavelet
    wav = ricker(np.arange(41) * par['dt'],f0=par['f0'])[0]
    # Parameters for events
    v = 1500
    ang_range = 50
    amp_range = 2
    i = 0
    amp_lim = 0.2
    t0 = [0.2,0.3,0.5,0.8]
    amp = [-0.5,1.2,-1.5,0.8]
    theta = [10,-10,5,-30]
    while i < samples: 
        # Making events
        mlin, mlinwav = linear2d(x, t, v, t0,theta, amp, wav)
        # Creating noise
        n = np.random.normal(loc=0,scale=0.25,size=(img_size,img_size))*random.uniform(-2,2)
        # Adding noise
        s = mlinwav
        ns = s+n
        clean_signal.append(s)
        noise.append(n)
        noisy_signal.append(ns)
        i +=1

    return  np.array(clean_signal).reshape(samples,img_size,img_size,1),np.array(noise).reshape(samples,img_size,img_size,1),np.array(noisy_signal).reshape(samples,img_size,img_size,1)




def PlotSeis(data, num=0, save=False):

    size = np.array(data[0]).shape[1]

        # Parameters for the seismic canvas
    par = {'ox':0, 'dx':12.5, 'nx':size, # offsets
       'ot':0, 'dt':0.004, 'nt':size, # time
       'f0': random.randint(5,30), 'nfmax': 50}
    
    # Make canvas
    t, t2, x, y = makeaxis(par)

    fig, axs = plt.subplots(1, len(data), figsize=(len(data*4), 7))

    vmin = -np.max(data[0][num])
    vmax = np.max(data[0][num])
    # Looping over datasets to compare
    for j in range(len(data)):
        im = axs[j].imshow(data[j][num].reshape(size,size).T, aspect='auto', interpolation='nearest',
            vmin=vmin, vmax=vmax, cmap='gray',
            extent=(x.min(), x.max(), t.max(), t.min())).set_cmap('Greys')

    # fig.colorbar(axs[-1], im)
    if save:
        file_name = input("file name:")
        plt.savefig('./results/images/%s_start%s.png'%(file_name,start))
