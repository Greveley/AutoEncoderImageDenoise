import numpy as np 
import matplotlib.pyplot as plt 
from pylops.utils.seismicevents import makeaxis, linear2d, hyperbolic2d
from pylops.utils.wavelets import ricker
import random
from skimage.util import random_noise

def MakeSeismic_VN(samples,img_size=256,num_events=10):

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
       'f0': random.randint(5,70), 'nfmax': 50}
       # initial tests, max freq was 50
    # Make canvas
    t, t2, x, y = makeaxis(par)
    # Make wavelet
    wav = ricker(np.arange(41) * par['dt'],f0=par['f0'])[0]
    # Parameters for events
    v = 1500
    # orig amp range was 50
    ang_range = 80
    amp_range = 2
    i = 0
    amp_lim = 0.8
    lv = 1500
    hv = 5000
    while i < samples: 
        iEv_l = 0
        iEv_h = 0
        t0_l = []
        t0_h = []
        theta_l = []
        amp_l = []
        amp_h = []
        vel_h = []
        num_lin = random.randint(2,num_events)
        num_hyp = num_events-num_lin
        while iEv_l <= num_lin:
            # Time of events
            t0_l.append(random.uniform(t.min(),t.max())*0.7) 
            # Angle of events
            theta_l.append(random.uniform(-ang_range,ang_range))
            # Amplitude of events
            amp_l.append(random.uniform(-amp_range,amp_range))
            # clipping events to be above -0.2 and 0.2 
            if amp_l[iEv_l]<0:
                amp_l[iEv_l] = np.min([-amp_lim,amp_l[iEv_l]])
            else: 
                amp_l[iEv_l] = np.max([amp_lim,amp_l[iEv_l]])
            iEv_l+=1
        while iEv_h <= num_hyp:
            # Time of events
            t0_h.append(random.uniform(t.min(),t.max())*0.7) 
            # Amplitude of events
            amp_h.append(random.uniform(-amp_range,amp_range))
            # velocity of hyperbolic events
            vel_h.append(random.uniform(lv,hv))
            # clipping events to be above -0.2 and 0.2 
            if amp_h[iEv_h]<0:
                amp_h[iEv_h] = np.min([-amp_lim,amp_h[iEv_h]])
            else: 
                amp_h[iEv_h] = np.max([amp_lim,amp_h[iEv_h]])
            iEv_h+=1
        
        # Making events
        mlin, mlinwav = linear2d(x, t, v, t0_l,theta_l, amp_l, wav)
        # print (t0_h, vel_h, amp_h)
        # Generate model
        m, mwav = hyperbolic2d(x, t, t0_h, vel_h, amp_h, wav)
        s = mwav+mlinwav

         # Creating and adding noise
        ns1 = random_noise(s,'speckle',clip=False,var=random.uniform(0.2,2))
        ns2 = random_noise(s,'gaussian',clip=False,var=random.uniform(0.05,0.5))
        ns3 = random_noise(s,'s&p',clip=False,amount=random.uniform(0.05,0.2))

        # Noise
        n1 = ns1 - s
        n2 = ns2 - s
        n3 = ns3 - s

        clean_signal.append(s)
        clean_signal.append(s)
        clean_signal.append(s)

        noise.append(n1)
        noise.append(n2)
        noise.append(n3)

        noisy_signal.append(ns1)
        noisy_signal.append(ns2)
        noisy_signal.append(ns3)

        i +=1

    return  np.array(clean_signal).reshape(samples*3,img_size,img_size,1),np.array(noise).reshape(samples*3,img_size,img_size,1),np.array(noisy_signal).reshape(samples*3,img_size,img_size,1)

def MakeSeismic_VN_dip(samples,img_size=256,num_events=10):

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
       'f0': random.randint(5,70), 'nfmax': 50}
       # initial tests, max freq was 50
    # Make canvas
    t, t2, x, y = makeaxis(par)
    # Make wavelet
    wav = ricker(np.arange(41) * par['dt'],f0=par['f0'])[0]
    # Parameters for events
    v = 1500
    # orig amp range was 50
    ang_range = 80
    amp_range = 2
    i = 0
    amp_lim = 0.8
    while i < samples: 
        iEv = 0
        t0 = []
        theta = []
        amp = []
        ang = random.uniform(-ang_range,ang_range)
        while iEv <= num_events:
            # Time of events
            t0.append(random.uniform(t.min(),t.max())*0.7) 
            # Angle of events
            theta.append(ang)
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
        s = mlinwav
         # Creating and adding noise
        ns1 = random_noise(s,'speckle',clip=False,var=random.uniform(0.2,2))
        ns2 = random_noise(s,'gaussian',clip=False,var=random.uniform(0.05,0.5))
        ns3 = random_noise(s,'s&p',clip=False,amount=random.uniform(0.05,0.2))

        # Noise
        n1 = ns1 - s
        n2 = ns2 - s
        n3 = ns3 - s

        clean_signal.append(s)
        clean_signal.append(s)
        clean_signal.append(s)

        noise.append(n1)
        noise.append(n2)
        noise.append(n3)

        noisy_signal.append(ns1)
        noisy_signal.append(ns2)
        noisy_signal.append(ns3)

        i +=1

    return  np.array(clean_signal).reshape(samples*3,img_size,img_size,1),np.array(noise).reshape(samples*3,img_size,img_size,1),np.array(noisy_signal).reshape(samples*3,img_size,img_size,1)


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
       'f0': random.randint(5,70), 'nfmax': 50}
       # initial tests, max freq was 30
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
