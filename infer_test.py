import numpy as np
from set_Defaults import *
from function import *
from fast_infer import *
from fast_learn_filters import *

def infer_test(model, y, F):

    #model = Init_param()
    layer = model.layer
    xdim = np.shape(y)[0]
    ydim = np.shape(y)[1]

    filter_size = model.filter_size[layer]
    num_feature_maps = model.num_feature_maps[layer]

    maxepochs = model.maxepochs[layer]
    # Number of iterations to run minimize
    min_iterations = model.min_iterations

    input_maps = model.num_input_maps[layer]

    lamb = model.lamb[layer]
    Binitial = model.Binitial
    Bmultiplier = model.Bmultiplier
    betaT = model.betaT
    beta = Binitial
    alphavalue = model.alphavalue[layer]
    kappa = model.kappa
    connectivity_matrix = model.conmat[layer]
    #  If the  z0 map is used at this layer.
    TRAIN_Z0 = model.TRAIN_Z0

    z = np.zeros(((xdim+filter_size-1), (ydim+filter_size-1), num_feature_maps, np.shape(y)[3]))
    w = z
    if(TRAIN_Z0):
        z0_filter_size = model.z0_filter_size
        psi = model.psi
        z0 = 0.01 * np.random.randn((xdim+z0_filter_size-1, ydim+z0_filter_size-1, input_maps, np.shapr(y)[3]))
    else:
        z0 = 0
        z0_filter_size = 1
        psi = 1
    for sample in range(np.shape(y)[3]):
        print("this is the %d image" % sample)
        zsample = z[:, :, :, sample]
        ysample = y[:, :, :, sample]
        wsample = w[:, :, :, sample]
        if TRAIN_Z0:
           z0sample = z0[:, :, :, sample]
        else:
           z0sample = 0
        for beta_iteration in range(betaT): # betaT=6
            if beta_iteration == 0:
                beta = Binitial # 1
            elif beta_iteration == betaT:
                print('.\n')
                beta = beta * Bmultiplier  # *10
            else:
                print('.')
                beta = beta * Bmultiplier
            print(beta)
            wsample[:] = solve_image(zsample, beta, alphavalue, kappa[layer])
            zsample[:] = fast_infer(min_iterations, zsample, wsample, ysample, F, z0sample, z0_filter_size, lamb,
                                         beta, connectivity_matrix, TRAIN_Z0)
            # F = fast_learn_filters(min_iterations, zsample, ysample, F, z0sample, z0_filter_size, lamb,
            #                                         connectivity_matrix, TRAIN_Z0)
            # F = plane_normalize(F)
            z[:, :, :, sample] = zsample
            w[:, :, :, sample] = wsample

    return z