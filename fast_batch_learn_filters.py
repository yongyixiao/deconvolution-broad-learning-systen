import numpy as np
from scipy import signal


"""
Updates the filters based on a batch of training samples
"""
def fast_batch_learn_filters(max_it, zbatch, ybatch, F, z0batch, z0_filter_size, lamb, C, TRAIN_Z0):

    sizeF = np.shape(F)
    num_feature_maps = np.shape(F)[3]
    num_input_maps = np.shape(F)[2]
    xdim = np.shape(ybatch)[0]
    ydim = np.shape(ybatch)[1]

    summed_F = np.zeros(sizeF)

    # Just simply loop over all the images in the batch.


    return 0