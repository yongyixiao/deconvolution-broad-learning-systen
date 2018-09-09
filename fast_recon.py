import numpy as np
from scipy import signal


"""
Reconstructs the input maps from the feature maps convolved with the filters
@:param z: the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps)
@:param F: the filters (Fxdim x Fydim x num_input_maps x num_feature_maps)
@:param C: the connectivity matrix for the layer.
@:param TRAIN_Z0 binary indicating if z0 should be used or not.
@:return I: the reconstructed input maps.
"""

def fast_recon(z0, z0_filter_size, z, F, C, TRAIN_Z0):
    num_feature_maps = np.shape(F)[3]
    num_input_maps = np.shape(F)[2]
    filter_size = np.shape(F)[0]
    xdim = np.shape(z)[0] - filter_size + 1
    ydim = np.shape(z)[1] - filter_size + 1
    I = np.zeros((xdim, ydim, num_input_maps))

    for j in range(num_input_maps):
        # Initialize a variable to keep the running some of the other convolutions between f*z
        convsum = np.zeros((xdim, ydim))

        # Loop over all the other filters and compute the sume of their convolutions (f*z)
        for k in range(num_feature_maps):
            if C[j, k] == 1: # Only do convolutions where connected.
                convsum = convsum + signal.convolve2d(z[:, :, k], F[:, :, j, k], mode='valid')

        I[:, :, j] = convsum[:, :]

    return I
