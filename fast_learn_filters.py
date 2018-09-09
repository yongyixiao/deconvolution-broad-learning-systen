import numpy as np
from scipy import signal


"""
# ###Updates the filters based on a single training sample
@:param max_it: number of conjugate gradient iterations
@:param z :the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps)
@:param y:the input maps for the layer (xdim * ydim * num_input_maps).
@:param F:the filters (Fxdim * Fydim * num_input_maps * num_feature_maps).
@:param z0: the z0 feature maps (may not be used)
@:param z0_filter_size:the size of the z0 filters
@:param lam: the coefficient on the reconstruction error term.
@:param C :the connectivity matrix for the layer.
@:param TRAIN_Z0: binary indicating if z0 should be used or not.
"""

def fast_learn_filters(max_it, z, y, F, z0, z0_filter_size, lam, C, TRAIN_Z0):
    # F[7,7,1,3]
    num_feature_maps = np.shape(F)[3]
    num_input_maps = np.shape(F)[2]
    xdim = np.shape(y)[0]
    ydim = np.shape(y)[1]

    # Initialize variable for the results
    conctemp = np.zeros(np.shape(F))
    new_F = np.reshape(F, (np.shape(F)[0]*np.shape(F)[1]*np.shape(F)[2]*np.shape(F)[3], 1))
    # ##Compute the right hand side (A'b) term Do the f'y convolutions.
    for j in range(num_input_maps):
        if TRAIN_Z0:
            z0conv = signal.convolve2d(z0[:, :, j], np.ones((z0_filter_size, z0_filter_size))/z0_filter_size,
                                       mode='valid')
        for k in range(num_feature_maps):
            if C[j, k] == 1:
                if TRAIN_Z0: # If using z0 maps, must convolve with the z0conv
                    conctemp[:, :, j, k] = signal.convolve2d(np.flipud(np.fliplr(z[:, :, k])), y[:, :, j], mode='valid')\
                                           - signal.convolve2d(np.flipud(np.fliplr(z[:, :, k])), z0conv, mode='valid')
                else:
                    conctemp[:, :, j, k] = signal.convolve2d(np.flipud(np.fliplr(z[:, :, k])), y[:, :, j], mode='valid')

    # This is the RHS. Only comput this once.
    new_conctemp = np.reshape(conctemp, (np.shape(F)[0]*np.shape(F)[1]*np.shape(F)[2]*np.shape(F)[3], 1)) #(7*7*3*9,1)
    Atb = lam * new_conctemp

    # ##Compute the left hand side (A'Ax) term Loop over each input plane.
    conctemp1 = np.zeros(np.shape(F))

    for j in range(num_input_maps):
        # Initialize a variable to keep the running some of the other convolutions between f*z.
        convsum = np.zeros((xdim, ydim))
        # Loop over all the other ks and compute the sume of their convolutions (f*z). This is the Ax term.
        for k in range(num_feature_maps):
            if C[j, k] == 1:
                convsum = convsum + signal.convolve2d(z[:, :, k], F[:, :, j, k], mode='valid')

        # This is the A'Ax term.
        for k in range(num_feature_maps):
            if C[j, k] == 1:
                conctemp1[:, :, j, k] = signal.convolve2d(np.flipud(np.fliplr(z[:, :, k])), convsum, mode='valid')

    # This is the left hand side.
    new_conctemp1 = np.reshape(conctemp1, (np.shape(F)[0]*np.shape(F)[1]*np.shape(F)[2]*np.shape(F)[3], 1)) #(7*7*3*9,1]
    AtAx = lam * new_conctemp1

    # Compute the residual.
    r = Atb - AtAx #[7*7*3*9,1]

    for itera in range(max_it):
        rho = np.dot(r.T, r) # 一个数
        if itera > 0:
            their_beta = rho / rho_1
            new_p = np.reshape(p, (np.shape(F)[0]*np.shape(F)[1]*np.shape(F)[2]*np.shape(F)[3], 1))
            new_p = r + their_beta * new_p
            p = np.reshape(new_p, np.shape(F))
        else:
            p = r
            p = np.reshape(p, np.shape(F))

        # ##Compute the left hand side (A'Ax) term Loop over each input map
        conctemp = np.zeros(np.shape(F))
        for j in range(num_input_maps):
            convsum = np.zeros((xdim, ydim))
            for k in range(num_feature_maps):
                if C[j, k] == 1:
                    convsum = convsum + signal.convolve2d(z[:, :, k], p[:, :, j, k], mode='valid')

            for k in range(num_feature_maps):
                if C[j, k] == 1:
                    conctemp[:, :, j, k] = signal.convolve2d(np.flipud(np.fliplr(z[:, :, k])), convsum, mode='valid')

        # This is the left hand side.
        new_conctemp = np.reshape(conctemp, (np.shape(F)[0]*np.shape(F)[1]*np.shape(F)[2]*np.shape(F)[3], 1)) #(7*7*3*9,1]
        q = lam * new_conctemp
        if itera == 0:
            new_p = np.reshape(p, (np.shape(F)[0]*np.shape(F)[1]*np.shape(F)[2]*np.shape(F)[3], 1))
        their_alpha = rho / np.dot(new_p.T, q)
        new_F = new_F + their_alpha * new_p  # update approximation vector
        r = r - their_alpha * q  # compute residual
        rho_1 = rho
    F = np.reshape(new_F, (np.shape(F)))
    return np.nan_to_num(F)