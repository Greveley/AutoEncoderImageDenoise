This repo contains code that uses a deep convolutional autoencoder to automatically detect and remove random noise from seismic images. 

# Methodology
The methodology consists of 2 stages:
## Stage 1: Synthetic Data Training
### Generate Data:
Synthetic seismograms of a random number of seismic events are generated using a ricker wavelet. Random characteristics of these seismic events include: 
* Frequency (between 10 and 70Hz) 
* Angle (beween +/- 50 degrees)
* Amplitude (between +/- 2)
These seismograms are the target output of the network. A random level of Gaussian noise is then added to the clean seismic data; this dataset is the input to the network. 
### Model generation: 
A number of models have been (and still are being) tested to give optimum results. These models include: 
* Convolutional autoencoders applied over windows of the input images.
* Unet architecture model. (so far this model gives the best results)
* 1D Artificial neural network.
* 1D Convolutional neural network. 
### Fitting models: 
These models are then fit to the synthetic data to effectively remove the random noise. Optimizer = SGD, loss function = MSE. 
	
## Stage 2: Real Data Training
The models trained on the synthetic data are then used as a starting point to train on real data. However, the ground truth clean seismic image is not know for the real seismic data. Therefore a psuedo-unsupervised training scheme is implemented by editing the loss function to minimise the correllation coefficient between the input image and the output image. 



The methodology brings together those found in two papers: 
*Saad, O. and Chen, Y., 2020. Deep denoising autoencoder for seismic random noise attenuation. GEOPHYSICS, 85(4), pp.V367-V376.
*Zhang, M., Liu, Y. and Chen, Y., 2019. Unsupervised Seismic Random Noise Attenuation Based on Deep Convolutional Neural Network. IEEE Access, 7, pp.179810-179822
