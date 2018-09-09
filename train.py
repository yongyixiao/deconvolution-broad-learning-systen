from set_Defaults import *
from createImages import *
from function import *
from train_recon_layer import *
from update_conmat import *
import pandas as pd
from load_mnist import *
import os
from scipy import signal
from BLS_demo import *
from infer_test import *
from loop_max_pool import *

def train():
    # 获得模型参数
    model = Init_param()

    model.conmat = update_conmat(model.num_layers, model.conmat_types, model.num_input_maps, model.num_feature_maps)
    print(model.conmat)
    if(not os.path.exists(model.fullsavepath)):
        os.makedirs(model.fullsavepath)

    """
    对于第一层：获得数据准备
    """
    # 如果使用z0，则覆盖对比度归一化为0
    if(model.TRAIN_Z0 == 1):
        model.CONTRAST_NORMALIZE = 0

    total_accuracy = []
    totaltest_accuracy = []
    epoch_num = 1
    for epocch in range(epoch_num):
        F_total = []
        print("第%d个batch" % epocch)
        # 获取图片
        # original_Image[96,96,3,10] ;mn[3,10],sd[3,10],xdim96,ydim96
        #original_images, xdim, ydim, mn, sd = createImages(model.fulldatapath, model.CONTRAST_NORMALIZE, model.ZERO_MEAN, model.COLOR_IMAGES, model.SQUARE_IMAGES)
        original_images, train_y, test_images, test_y, xdim, ydim, mn, sd = loadImages(model.CONTRAST_NORMALIZE, model.ZERO_MEAN, model.COLOR_IMAGES, model.SQUARE_IMAGES)

        #test_images, test_y , test_xdim, test_ydim, test_mn, test_sd = loadImages(model.CONTRAST_NORMALIZE, model.ZERO_MEAN, model.COLOR_IMAGES, model.SQUARE_IMAGES)
        # 将输出归一化[-1,1]
        original_images = rescale_all_n1_1(original_images)
        test_images = rescale_all_n1_1(test_images)
        # Store the original image sizes in each model
        model.orixdim = np.shape(original_images)[0]
        model.oriydim = np.shape(original_images)[1]

        # Make sure the first layer input maps make sense (regarless of color at this point)
        model.num_input_maps[0] = np.shape(original_images)[2]
        print(model.oriydim)

        save = 1
        #if save:
        # For the first layer the input maps are the original images.
        y = original_images
        contemp = test_images

        # ##Normalize the data if needed
        # If pooling is required, apply it to the feature maps (y)
        if operator.eq(model.norm_types[0], 'Max'):
            pool_map, pooled_indices = max_pool(original_images, model.norm_sizes[0])
            test_map, test_indices = max_pool(test_images, model.norm_sizes[0])
            y = pool_map
            model.pooled_indices0 = pooled_indices
            contemp = test_map
        # Loop over each layer you want to train.
        # model.num_layers = 2
        modelargs = ''
        for train_layer in range(model.num_layers):
            model.layer = train_layer

            #if epocch == 0:
            F = np.random.randn(7, 7, model.num_input_maps[train_layer], model.num_feature_maps[train_layer])#[7,7,1,3],[7,7,9,45]
            # Normalize the planes of F
            F = plane_normalize(F)
            F_total.append(F)

            # don't use z0 maps in higher layers
            if(train_layer > 0):
                model.TRAIN_Z0 = 0
            # xdim and ydim are the size of the input layers which are vectorized into
            print("Training Layer %d of a %d-Layer Model" % (model.layer+1, model.num_layers))
            print("Number of Input Maps = %d, Number of Feature Maps = %d" % (model.num_input_maps[train_layer], model.num_feature_maps[train_layer]))

            #  A string of parameters to pass to each layer
            if train_layer == 0:
                stop = 0
            else:
                stop = 1
            for layer in range(train_layer+1, stop, -1):
                modelargs = modelargs + ',' + 'model' + str(layer)
                modelargs = modelargs + ',' +  'F' + str(layer)
                modelargs = modelargs + ',' + 'z0' + str(layer)
                modelargs = modelargs + ',' + 'pooled_indices' + str(layer)
            # get rid of the first ',' that is in the string
            print(modelargs)
            modelargs = modelargs[1:]
            # Add the input_map (y), original images and
            modelargs = modelargs + ',pooled_indices0,y,original_images'
            print(modelargs)
            #for i in range(3):
            F_new, y_new, cost_energy = train_recon_layer(model, modelargs, y, original_images, F_total)
            F_total[train_layer] = F_new

            # Normalization Procedure
            print("这得到最终训练集的feature")
            y_new = infer_test(model, y, F_total[train_layer])
            if operator.eq(model.norm_types[train_layer+1], 'Max'):
                pooled_maps, pooled_indices1 = max_pool(y_new, model.norm_sizes[train_layer+1])
                y_new = pooled_maps
                model.pooled_indices1 = pooled_indices1
            y = rescale_all_n1_1(y_new) #[30,30,3,100]

            print("加载测试集的feature")
            contemp = infer_test(model, contemp, F_total[train_layer])
            if operator.eq(model.norm_types[train_layer+1], 'Max'):
                pooled_maps, pooled_testindices1 = max_pool(contemp, model.norm_sizes[train_layer+1])
                contemp = pooled_maps
            contemp = rescale_all_n1_1(contemp)
        train_x = np.reshape(y, (np.shape(y)[0]*np.shape(y)[1]*np.shape(y)[2], np.shape(y)[3]))
        train_x = train_x.T
        contemp = np.reshape(contemp, (np.shape(contemp)[0]*np.shape(contemp)[1]*np.shape(contemp)[2], np.shape(contemp)[3]))
        contemp = contemp.T
        train_accuracy, test_accuracy = bls_no(train_x, train_y, contemp, test_y)
        total_accuracy.append(train_accuracy)
        totaltest_accuracy.append(test_accuracy)

    print("训练集准确率为：")
    for i in range(epoch_num):
        print(total_accuracy[i])
    print("测试集准确率为")
    for i in range(epoch_num):
        print(totaltest_accuracy[i])
train()


