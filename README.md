This repo contains code that uses a deep convolutional autoencoder to automatically detect and remove random noise from seismic images. 

The methodology consists of 2 stages:

***Stage 1***

  Generate Data:
  Synthetic seismograms of a random number of seismic events are generated using a ricker wavelet. Random characteristics of these seismic events include: 
    - Frequency (between 10 and 70Hz) 
    - Angle (beween +/- 50 degrees)
    - Amplitude (between +/- 2)
 
![plot](https://github.com/Greveley/AutoEncoderImageDenoise/files/6191660/Synth.pdf,raw=True)
	 
  These seismograms are the target output of the network. A random level of Gaussian noise is then added to the clean seismic data; this dataset is the input to the network. 
	
  Model generation: 
	A number of models have been (and still are being) tested to give optimum results. These models include: 
	- Convolutional autoencoders applied over windows of the input images.
	- Unet architecture model. 
	- 1D Artificial neural network.
	- 1D Convolutional neural network. 

  Fitting models: 
	
	
  



The methodology brings together those found in two papers: 

  Saad, O. and Chen, Y., 2020. Deep denoising autoencoder for seismic random noise attenuation. GEOPHYSICS, 85(4), pp.V367-V376.
  Zhang, M., Liu, Y. and Chen, Y., 2019. Unsupervised Seismic Random Noise Attenuation Based on Deep Convolutional Neural Network. IEEE Access, 7, pp.179810-179822
