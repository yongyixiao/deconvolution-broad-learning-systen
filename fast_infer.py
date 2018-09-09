import numpy as np
from scipy import signal
import numpy.linalg as nlg

"""
Updates the feature maps for a single training sample (image),This is done via
 conjuagte gradient.
 @:param max_it: number of conjugate gradient iterations
 @:param z: the feature maps to update(xdim+filter_size x ydim+filter_size x num_feature_maps)
 @:param w: the auxilary variable (same size as z).
 @:param y: the input maps for the layer(xdim * ydim * num_input_maps)
 @:param F: the filters (Fxdim x Fydim x num_input_maps x num_feature_maps)
 @:param z0: the z0 feature maps (may not be used)
 @:param  z0_filter_size: the size of the z0 filters (if used)
 @:param lam: the coefficient on the reconstruction error term.
 @:param beta: the continuation variable on the ||z-x|| term.
 @:param C: the connectivity matrix for the layer.
 @:param  TRAIN_Z0: binary indicating if z0 should be used or not.
 @:return  z: the updated feature maps.
"""

def fast_infer(max_it, z, w, y, F, z0, z0_filter_size, lam, beta, C, TRAIN_Z0):

    # get the number  # F[7,7,1,3] [7,7,3,9]
    num_feature_maps = np.shape(F)[3]
    num_input_maps = np.shape(F)[2]
    xdim = np.shape(y)[0]
    ydim = np.shape(y)[1]

    # Initialize the running sum for each feature map
    conctemp = np.zeros(np.shape(z))
    # new_z = np.reshape(z, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2], 1))
    # ## Compute the right hand side (A'b) term Do the f'y convolutions.
    for j in range(num_input_maps):
        if TRAIN_Z0:
            z0conv2 = signal.convolve2d(z0[:, :, j], np.ones((z0_filter_size, z0_filter_size))/z0_filter_size, mode='valid')
        for k in range(num_feature_maps):
            if C[j, k] == 1:
                if TRAIN_Z0:
                    conctemp[:, :, k] = conctemp[:, :, k] + signal.convolve2d(y[:, :, j], np.flipud(np.fliplr(F[:, :, j, k])), mode='full')\
                                     - signal.convolve2d(z0conv2, np.flipud(np.fliplr(F[:, :, j, k])), mode='full')
                else:
                    conctemp[:, :, k] = conctemp[:, :, k] + signal.convolve2d(y[:, :, j], np.flipud(np.fliplr(F[:, :, j, k])), mode='full')
    # This is the RHS. Only compute this once
    new_conctemp = np.reshape(conctemp, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2], 1))
    new_w = np.reshape(w, (np.shape(w)[0]*np.shape(w)[1]*np.shape(w)[2], 1))
    Atb = lam * new_conctemp + beta * new_w  # [30*30*3,1]

    # ## Compute the left hand side (A'Ax) term
    # Initialize the running sum for each feature map
    conctemp1 = np.zeros(np.shape(z))
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
                #  Place in correct location so when conctemp(:) is used below it will be the correct vectorized form for dfz.
                conctemp1[:, :, k] = conctemp1[:, :, k] + signal.convolve2d(convsum, np.flipud(np.fliplr(F[:, :, j, k])),
                                                                          mode='full')
    #  This is the left hand side.
    new_conctemp1 = np.reshape(conctemp1, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2], 1))
    new_z = np.reshape(z, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2], 1))
    AtAx = lam * new_conctemp1 + beta * new_z # [30*30*3,1]


    # Compute the residual
    r = Atb - AtAx  # [30*30*3,1]

    for i in range(max_it): # 2
        rho = np.dot(r.T, r) # 一个数

        if i > 0:
            # rho1_new = np.double(np.abs(rho_1) > 1e-9) * rho
            # rho1_inv = nlg.inv(rho_1)
            # their_beta = np.dot(rho1_new, rho1_inv)
            # p[:] = r[:] + np.dot(their_beta, p[:])
            their_beta = np.double(np.abs(rho_1) > 1e-9) * rho / rho_1
            new_p = np.reshape(p, (np.shape(w)[0]*np.shape(w)[1]*np.shape(w)[2], 1))
            new_p = r + their_beta * new_p
            p = np.reshape(new_p, np.shape(w))
        else:
            p = r
            p = np.reshape(p, np.shape(w)) # [30,30,3]

        # ## Compute the left hand side (A'Ax) term
        # Initialize the running sum for each feature map.
        conctemp = np.zeros(np.shape(z))
        for j in range(num_input_maps):
            convsum = np.zeros((xdim, ydim))
            for k in range(num_feature_maps):
                if C[j, k] == 1:
                    convsum = convsum + signal.convolve2d(p[:, :, k], F[:, :, j, k], mode='valid')

            # this is the A'Ax term
            for k in range(num_feature_maps):
                if C[j, k] == 1:
                    conctemp[:, :, k] = conctemp[:, :, k] + signal.convolve2d(convsum, np.flipud(np.fliplr(F[:, :, j, k])),
                                                                              mode='full')
        # This is the left hand side.
        new_conctemp = np.reshape(conctemp, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2], 1))

        new_p = np.reshape(p, (np.shape(w)[0]*np.shape(w)[1]*np.shape(w)[2], 1))
        q = lam * new_conctemp + beta * new_p
        temp = np.dot(new_p.T, q) # 一个数
        temp_new = np.double(np.abs(temp) >= 1e-9)
        rho_new = temp_new * rho
        their_alpha = rho_new / temp
        # update approximation vector
        new_z = new_z + new_p * their_alpha
        # compute residual
        r = r - their_alpha * q
        rho_1 = rho
        #print("这是their_alpha %f" % their_alpha)
    z = np.reshape(new_z, (np.shape(z)[0], np.shape(z)[1], np.shape(z)[2]))
    return np.nan_to_num(z)