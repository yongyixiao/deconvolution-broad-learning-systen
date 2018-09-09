
from set_Defaults import *
from function import *
from fast_infer import *
from fast_learn_filters import *
from fast_recon import *
from loop_max_pool import *
import matplotlib.pyplot as plt
import operator
import numpy as np
"""
This is a combined function to both train or reconstruct images in the pixel space.
Reconstructs a single layer (specified by model.layer) of the model
based on the input y. y is the input maps of model.layer which maybe be
the feature maps of a layer below. The feature maps are inferred from
this and then used to reconstruct a y' reconstruction of the input maps.
Inputs: The inputs are passed in varargin (a cell array) because when
12rs, all the lower layers need to be passed in to reconstruct down to the lowest level.
"""

"""
@:param varargin should be of the following format: (example for 2 layers)
        model2: struct with parameters for top layer
        F2: third layer filters or []
        z02: third layer z0 maps or []
        pooled_inpdices2: the indices from Max pooling after L2 (usually not every used) to allow reconstructions
        model1: struct with parameters for layer
        F1: layer 1 filters
        z02: fist layer z0 maps or []
        y :the input maps for the given layer (may be noisy if denoising)
        original_images: the clean images
@:return  F: the learned (or previous if reconstructing) filters
@:return  z: the inferred feature maps.
@:return  z0: the inferred z0 feature maps (or [] if not used).
@:return  recon_images: the reconstructed images
@:return  model: some fields of the model structure will be updated within (with
xdim, ydim, and errors for example)
"""

def train_recon_layer(model, varargin, y,original_images, F):
    #F = []
    #model = Init_param()
    # Read in the variables arguments
    # The layer to be inferred is the first on input.
    layer = model.layer
    y = y  # 池化后的图片。若无池化，则为原图片 #[24,24,1,100]
    original_images = original_images   # 原图片

    xdim = np.shape(y)[0]
    ydim = np.shape(y)[1]

    model.num_input_maps[layer] = np.shape(y)[2]
    input_maps = model.num_input_maps[layer]

    # Parameters. DO NOT CHANGE HERE.
    # Size of each filter
    filter_size = model.filter_size[layer]
    # Number of filters to learn
    num_feature_maps = model.num_feature_maps[layer]

    # Number of epochs total through the training set
    maxepochs = model.maxepochs[layer]
    # Number of iterations to run minimize
    min_iterations = model.min_iterations

    # Sparsity Weighting
    lamb = model.lamb[layer]
    UPDATE_INPUT = model.UPDATE_INPUT
    lambda_input = model.lambda_input

    # Use 0.01, batch, and put lambda = lambda + 0.1 to see that the first filters just take patches.
    # Dummy regularization weighting. This starts at beta=Binitial, increases
    # by beta*Bmultiplier each iteration until T iterations are done.
    Binitial = model.Binitial
    Bmultiplier = model.Bmultiplier
    betaT = model.betaT
    beta = Binitial
    # The alpha-norm on the dummy regularizer variables
    alphavalue = model.alphavalue[layer]
    kappa = model.kappa

    # Connectivity Matrix, this is a input_maps by num_feature_maps matrix.
    connectivity_matrix = model.conmat
    #  If the  z0 map is used at this layer.
    TRAIN_Z0 = model.TRAIN_Z0

    # If this is set to 1, then randomize the order in which the images are
    # selecdted from the training set. Random order changes at each epoch.
    # If this is set to 0, then they are picked in order.
    RANDOM_IMAGE_ORDER = model.RANDOM_IMAGE_ORDER
    # Fitlers are set after each batch while z's are every image sample.
    UPDATE_FILTERS_IN_BATCH = model.UPDATE_FILTERS_IN_BATCH
    # No point in batching data when not updating the filters in batch.
    if(UPDATE_FILTERS_IN_BATCH == 0):
        batch_size = np.shape(y)[3]  # 100
    else:
        batch_size = model.batch_size

    # Set this to 1 if you want to store the Filter matrices
    SAVE_RESULTS = model.SAVE_RESULTS
    fullsavepath = model.fullsavepath
    PLOT_RESULTS = model.PLOT_RESULTS
    train_recon = ''
    if(operator.eq(model.expfile, 'train')):
        exfile = model.expfile
        train_recon = 'train'
    elif(operator.eq(model.expfile, 'train_remaining_layers')):
        exfile = 'train'
        train_recon = 'train'
    elif(operator.eq(model.expfile, 'recon')):
        expfile = model.expfile
        train_recon = 'recon'

    if(operator.eq(train_recon, 'recon')):
        DISPLAY_ERRORS = 1
    else:
        DISPLAY_ERRORS = model.DISPLAY_ERRORS

    #### Preprocessing of the images

    # This calculates the number of mini-batches, this will be 1.
    number_of_batches = max(np.shape(y)[3] // batch_size, 1)

    y_input = y
    noisy_SNR = np.zeros((1, np.shape(y)[3]))
    # Compute SNR values again noisy image (y) and original image
    # if(layer == 1):
    #     if(operator.eq(model.norm_types[0], 'None')):
    #         # Check if the input image was normalized.
    #         for i in range(np.shape(y)[3]): # for each images
    #             # 求两张图片的信噪比
    #             noisy_SNR[i] = compute_snp()


    ### Plot the input images (or maps for higher layers).
    if(PLOT_RESULTS > 0):

        if(input_maps == 1 or input_maps == 3):
            sdispims(y)
        else:
            sdispmaps(y)

    ### TRAINING PHASE SETUP
    print("nihao train")
    # if(operator.eq(train_recon,'train')):
    #     # Initialize the F matrix to random number (0,1)
    #     # This F matrix stores all the filters as columns
    #     # There are filter_size*filter_size by input_maps by num_feature_maps
    #     F = np.random.randn(filter_size, filter_size, input_maps, num_feature_maps)#[7,7,1,3],[7,7,9,45]
    #     # Normalize the planes of F
    #     F = plane_normalize(F)
    #     print(np.shape(F))
    #
    # F = F

    # Initialize the feature maps to be random values [30,30,3,100]
    z = np.zeros(((xdim+filter_size-1), (ydim+filter_size-1), num_feature_maps, np.shape(y)[3]))

    # If using the z0 for the given layer.
    if(TRAIN_Z0):
        z0_filter_size = model.z0_filter_size
        psi = model.psi
        z0 = 0.01 * np.random.randn((xdim+z0_filter_size-1, ydim+z0_filter_size-1, input_maps, np.shapr(y)[3]))
    else:
        z0 = 0
        z0_filter_size = 1
        psi = 1

    # Introduce the dummy variables w, same size as z [30,30,3,100]
    w = z

    # Initialize a matrix to store the reconstructed images.
    recon_images = np.zeros(np.shape(original_images))

    # ##Loop through the entire training set this number of times.

    # Clear the error matrices (especially for reconstruction)
    model_dim = np.shape(y)[3]
    model.update_noise_rec_error = np.zeros((maxepochs, model_dim))
    model.pix_noise_rec_error = np.zeros((maxepochs, model_dim))
    model.pix_clean_rec_error = np.zeros((maxepochs, model_dim))
    model.pix_clean_SNR_error = np.zeros((maxepochs, model_dim))
    model.pix_update_rec_error = np.zeros((maxepochs, model_dim))
    model.reg_error = np.zeros((maxepochs, model_dim))
    model.beta_rec_error = np.zeros((maxepochs, model_dim))
    model.unscaled_total_energy = np.zeros((maxepochs, model_dim))
    model.scaled_total_energy = np.zeros((maxepochs, model_dim))

    feature = []
    total_filter = []
    safe_energy = []
    for epoch in range(maxepochs):
        print("这是第%d次epoch" % epoch)
        cost_energy = 0
        permbatchindex = []
        if RANDOM_IMAGE_ORDER :
            permbatchindex = generaterand(permbatchindex, 0, np.shape(y)[3]-1)
        else:
            for i in range(np.shape(y)[3]):
                permbatchindex.append(i+1)
        total_image_num = 0
        # batch_size = 10 ,number_of_batches=1
        for batch in range(number_of_batches):
            #  Get the start and end index for the given batch
            start_of_batch = batch * batch_size + 1
            # The last batch should contain all the remaining images
            if(batch+1 == number_of_batches):
                end_of_batch = len(permbatchindex)
            else:
                end_of_batch = (batch+1) * batch_size
            batch_indices = permbatchindex[int(start_of_batch-1):int(end_of_batch)] #[1,10]
            # Keep track of the errors for the whole batch.
            image_num = 0

            # ##Go through each image in the batch.
            for sample in range(len(batch_indices)):
                #sample = batch_indices[index]
                # ##Setups each sample's variables
                # Count the number of images processed.
                image_num = image_num + 1
                total_image_num = total_image_num + 1

                #  Get only those values for the current training sample.
                zsample = z[:, :, :, sample] # [30.30,num_feature_maps(3)] [36,36,9]
                ysample = y[:, :, :, sample] #[24,24,1]  [30,30,3]
                wsample = w[:, :, :, sample]
                if TRAIN_Z0:
                    z0sample = z0[:, :, :, sample]
                else:
                    z0sample = 0
                if DISPLAY_ERRORS == 0:
                    print("layer :%d,epoch :%d, batch:%d,image:%d (%d/%d)" % (model.layer, epoch, batch, sample, total_image_num, np.shape(y)[3]))
                elif np.mod(epoch, DISPLAY_ERRORS) != 0:
                    print("layer :%d,epoch :%d, batch:%d,image:%d (%d/%d)" % (model.layer, epoch, batch, sample, total_image_num, np.shape(y)[3]))
                else:
                    print("layer :%d,epoch :%d, batch:%d,image:%d (%d/%d)" % (model.layer, epoch, batch, sample, total_image_num, np.shape(y)[3]))

                # ##Beta Regeme
                for beta_iteration in range(betaT): # betaT=6
                    if beta_iteration == 0:
                        beta = Binitial # 1
                    elif beta_iteration == betaT:
                        print('.\n')
                        beta = beta * Bmultiplier  # *10
                    else:
                        print('.')
                        beta = beta * Bmultiplier
                    # ##Update the w values based on the current sample of z.
                    wsample[:] = solve_image(zsample, beta, alphavalue, kappa[layer])
                    # [102,102,9]

                    # ##Update Feature Maps
                    zsample[:] = fast_infer(min_iterations, zsample, wsample, ysample, F[layer], z0sample, z0_filter_size, lamb,
                                         beta, connectivity_matrix[layer], TRAIN_Z0)

                # ###Subsample and Inverse Subsample Feature Maps
                if model.SUBSAMPLING_UPDATES:
                    if model.norm_types[layer+1] == 'Max':
                        print("Max pooling and unpooling")
                        zsample = loop_max_pool(zsample, model.norm_sizes[layer+1])

                print("update filters")
                # ## Update Filters
                if operator.eq(train_recon, 'train'):
                    if UPDATE_FILTERS_IN_BATCH == 0:
                        # if (model.CONTRAST_NORMALIZE == 1 or model.COLOR_IMAGES == 0) and model.layer == 1:
                        #     filter_min = min_iterations
                        #     model.min_iterations  = filter_min
                        # else:
                        #     filter_min = min_iterations * 2
                        #     model.min_iterations  = filter_min
                        F[layer] = fast_learn_filters(min_iterations, zsample, ysample, F[layer], z0sample, z0_filter_size, lamb,
                                                    connectivity_matrix[layer], TRAIN_Z0)
                        F[layer] = plane_normalize(F[layer])
                # ## Copy the updated z's for the sample back into the z and w.
                z[:, :, :, sample] = zsample
                w[:, :, :, sample] = wsample
                if TRAIN_Z0:
                    z0sample[:, :, :, sample] = z0sample

                # 重构的feature map
                if layer == 0:
                    if operator.eq(model.norm_types[layer+1], 'Max'):
                        recon_z1, pooled_indices1 = max_pool(zsample, model.norm_sizes[layer+1])
                    elif operator.eq(model.norm_types[layer+1], 'None'):
                        recon_z1 = zsample
                    if operator.eq(model.norm_types[layer+1], 'Max'):
                        recon_z1 = reverse_max_pool(recon_z1, pooled_indices1[:, :, :], model.norm_sizes[1], [xdim + filter_size - 1, ydim + filter_size - 1])
                    recon_z0 = fast_recon(z0, z0_filter_size, recon_z1, F[layer], connectivity_matrix[layer], TRAIN_Z0)
                    if operator.eq(model.norm_types[0], 'Max'):
                        recon_z0 = reverse_max_pool(recon_z0, model.pooled_indices0[:, :, :, sample], model.norm_sizes[0], [model.orixdim, model.oriydim])
                    recon_images[:, :, :, sample] = recon_z0
                elif layer == 1:
                    recon_z1 = zsample
                    recon_z0 = fast_recon(z0, z0_filter_size, recon_z1, F[layer], connectivity_matrix[layer], TRAIN_Z0)
                    if operator.eq(model.norm_types[layer], 'Max'):
                       reverse_recon_z0 = reverse_max_pool(recon_z0, model.pooled_indices1[:, :, :, sample], model.norm_sizes[1], [np.shape(recon_z0)[0]*2, np.shape(recon_z0)[1]*2])
                    new_recon_z0 = fast_recon(z0, z0_filter_size, reverse_recon_z0, F[layer-1], connectivity_matrix[layer-1], TRAIN_Z0)
                    if operator.eq(model.norm_types[layer], 'Max'):
                        new_recon_z0 = reverse_max_pool(new_recon_z0, model.pooled_indices0[:, :, :, sample], model.norm_sizes[0], [model.orixdim, model.oriydim])
                    recon_images[:, :, :, sample] = new_recon_z0

                if layer == 0:
                    # Compute error versus the input maps (recon_z# where # is the layer below.
                    a, b = max_pool(recon_z0, model.norm_sizes[layer])
                    recon_error1 = np.sqrt(np.sum((a - y_input[:, :, :, sample]) ** 2))
                    model.pix_noise_rec_error[epoch, sample] = recon_error1
                    # Layer 1's error versus the updated pixel space images.
                    recon_error2 = np.sqrt(np.sum((a - y[:, :, :, sample]) ** 2))
                    model.pix_update_rec_error[epoch, sample] = recon_error2

                    # Updated y versus the noisy input reconstruction error
                    upd_rec_error = np.sqrt(np.sum((y_input[:, :, :, sample] - y[:, :, :, sample]) ** 2))
                    model.update_noise_rec_error[epoch, sample] = upd_rec_error

                    # Compute regularization error.
                    new_zsample = np.reshape(zsample, (np.shape(zsample)[0]*np.shape(zsample)[1]*np.shape(zsample)[2], 1))
                    reg_error = np.sum(np.abs(new_zsample))
                    model.reg_error[epoch, sample] = reg_error

                    # Compute Beta reconstruction error
                    beta_error = np.sqrt(np.sum((w[:, :, :, sample] - z[:, :, :, sample]) ** 2))
                    model.beta_rec_error[epoch, sample] = beta_error

                    # Layer 1's error versus the clean pixel space images.
                    recon_error = np.sqrt(np.sum((recon_images[:, :, :, sample] - original_images[:, :, :, sample]) ** 2))
                    print(":::pix clean total error: %f, recon error: %f, reg_error: %f" %  (recon_error + reg_error, recon_error, reg_error))
                    model.pix_clean_rec_error[epoch, sample] = recon_error

                    # Compute the sum of each term (with coefficients)
                    model.scaled_total_energy[epoch, sample] = lamb / 2 * model.pix_update_rec_error[epoch, sample] \
                                                                  + kappa[layer] * model.reg_error[epoch, sample] + lambda_input / 2 * upd_rec_error + (beta/2)/kappa[layer] * model.beta_rec_error[epoch, sample]
                    print(":::scaled Energy F Total: %f, Lay reg: %f, update error :%f, lay rec Upd: %f, lay rec Beta: %f" %
                          (model.scaled_total_energy[epoch, sample], kappa[layer]*model.reg_error[epoch, sample],
                           lambda_input*upd_rec_error, lamb/2*model.pix_update_rec_error[epoch, sample], (beta/2)*model.beta_rec_error[epoch, sample]))

                if layer == 1:
                    # Compute error versus the input maps (recon_z# where # is the layer below.
                    recon_error1 = np.sqrt(np.sum((recon_z0 - y_input[:, :, :, sample]) ** 2))
                    model.pix_noise_rec_error[epoch, sample] = recon_error1
                    # Layer 1's error versus the updated pixel space images.
                    recon_error2 = np.sqrt(np.sum((recon_z0 - y[:, :, :, sample]) ** 2))
                    model.pix_update_rec_error[epoch, sample] = recon_error2

                    # Updated y versus the noisy input reconstruction error
                    upd_rec_error = np.sqrt(np.sum((y_input[:, :, :, sample] - y[:, :, :, sample]) ** 2))
                    model.update_noise_rec_error[epoch, sample] = upd_rec_error

                    # Compute regularization error.
                    new_zsample = np.reshape(zsample, (np.shape(zsample)[0]*np.shape(zsample)[1]*np.shape(zsample)[2], 1))
                    reg_error = np.sum(np.abs(new_zsample))
                    model.reg_error[epoch, sample] = reg_error

                    # Compute Beta reconstruction error
                    beta_error = np.sqrt(np.sum((w[:, :, :, sample] - z[:, :, :, sample]) ** 2))
                    model.beta_rec_error[epoch, sample] = beta_error

                    # Layer 1's error versus the clean pixel space images.
                    recon_error = np.sqrt(np.sum((recon_images[:, :, :, sample] - original_images[:, :, :, sample]) ** 2))
                    print(":::pix clean total error: %f, recon error: %f, reg_error: %f" %  (recon_error + reg_error, recon_error, reg_error))
                    model.pix_clean_rec_error[epoch, sample] = recon_error

                    # Compute the sum of each term (with coefficients)
                    model.scaled_total_energy[epoch, sample] = lamb / 2 * model.pix_update_rec_error[epoch, sample] \
                                                                  + kappa[layer] * model.reg_error[epoch, sample] + lambda_input / 2 * upd_rec_error + (beta/2)/kappa[layer] * model.beta_rec_error[epoch, sample]
                    print(":::scaled Energy F Total: %f, Lay reg: %f, update error :%f, lay rec Upd: %f, lay rec Beta: %f" %
                          (model.scaled_total_energy[epoch, sample], kappa[layer]*model.reg_error[epoch, sample],
                           lambda_input*upd_rec_error, lamb/2*model.pix_update_rec_error[epoch, sample], (beta/2)*model.beta_rec_error[epoch, sample]))


                    total_energy = model.scaled_total_energy[epoch, sample]

            # if operator.eq(train_recon, 'train'):
            #     if UPDATE_FILTERS_IN_BATCH:
            #         print("updating filters in batch")
            #         if TRAIN_Z0:
            #             z0batch = z0[:, :, :, batch_indices]
            #         else:
            #             z0batch = []
            #         F = fast_batch_learn_filters(min_iterations, z[:, :, :, batch_indices], y[:, :, :, batch_indices],
            #                                      F, z0batch, z0_filter_size, lamb, connectivity_matrix, TRAIN_Z0)
            #         F = plane_normalize(F)

            # 每次epoch的所以照片的损失代价
            # Compute the average errors over the batch
            if epoch % DISPLAY_ERRORS == 0 and layer == 0:
                print("Layer:%d, Epoch:%d, Batch:" % (layer, epoch))
                clean_error = np.mean(model.pix_clean_rec_error[epoch, :]) + np.mean(model.reg_error[epoch, :])
                print("::: Pix Clean Total error:%f, Recon error:%f, Reg error:%f" %
                      (clean_error, np.mean(model.pix_clean_rec_error[epoch, :]), np.mean(model.reg_error[epoch, :])))
                scale_energy = lamb/2*np.mean(model.pix_update_rec_error[epoch, :]) + kappa[layer]*np.mean(model.reg_error[epoch, :]) \
                               + lambda_input*np.mean(model.update_noise_rec_error[epoch, :]) + (beta/2)*np.mean(model.beta_rec_error[epoch, :])
                print("::: Scaled Energy F Total:%f, Lay Reg:%f, Update Error:%f, Lay Rec Upd:%f, Lay Rec Beta:%f " %
                      (scale_energy, kappa[layer]*np.mean(model.reg_error[epoch, :]), lambda_input/2*np.mean(model.update_noise_rec_error[epoch, :]),
                       lamb/2*np.mean(model.pix_update_rec_error[epoch, :]), (beta/2)*np.mean(model.beta_rec_error[epoch, :])))
            #safe_energy.append(scale_energy)

    # x_dim = []
    # for i in range(maxepochs):
    #     x_dim.append(i+1)
    # plt.figure('cost_function')
    # plt.plot(x_dim, safe_energy, 'r')
    # plt.xlabel('epoch')
    # plt.ylabel('cost')
    # plt.show()

    return F[layer], z, safe_energy
