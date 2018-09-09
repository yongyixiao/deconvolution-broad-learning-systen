from set_Defaults import *
import os
import numpy as np
from PIL import Image
import operator
from scipy import signal
import matplotlib.pyplot as plt

"""
这将获取输入文件夹中的所有图像，将它们转换为所需的颜色空间，通过标准偏差（如果需要）消除平均值/除法，
并且对比度归一化图像（如果需要）。 如果图像大小不同，
则会用零填充它们（对比度归一化后）以使它们成方形（假定它们的所有图像具有相同的最大尺寸）
"""

"""
生成一个二维高斯核函数
"""
def gaussian_2d_kernal(kersize, sigma):
    kernal = np.zeros([kersize, kersize])
    center = kersize//2
    if sigma == 0:
       sigma = ((kersize-1)*0.5-1)*0.3+0.8
    s = (sigma**2)*2
    sum_val = 0
    for i in range(kersize):
        for j in range(kersize):
            x = i - center
            y = j - center
            kernal[i, j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernal[i, j]
    sum_val = 1/sum_val
    aa = kernal * sum_val
    return aa


"""
@:param img_path:图片文件路径
@:param CONTRAST_NORMAILZE：是否对图片对比度归一化 默认为：1
@:param ZERO_MEAN:是否减去均值再除以标准差  默认为：1
@:param COLOR_TYPE:a string of: 'gray','rgb','ycbcr','hsv'. Defaults to 'rgb'
@:param SQUARE_IMAGES:是否对图像进行平方，默认为：0

@:return I 图片：xdim * ydim * color_channal * numimages
@:return mn 均值。若ZERO_MEAN设置
@:return sd 标准差
@:return xdim 图片x方向的size
@:return ydim 图片y方向的size
@:return resI 对比度归一化图像，若CONTRAST_NORMAILZE设置

"""
def createImages(img_path,CONTRAST_NORMAILZE,ZERO_MEAN,COLOR_TYPE,SQUARE_IMAGES):

    image_num = 0
    # 文件里的图片数量
    image_total = []
    image_size = len([filename for filename in os.listdir(img_path) if
                      os.path.isfile(os.path.join(img_path, filename))])
    if(np.equal(image_size,0)):
        print("No Images in this directory")
    for filename in os.listdir(img_path):
        image_total.append(filename)
    # 通过文件里图片循环
    orig_I = []

    for file in range(image_size):
        print("loading : " + image_total[file])
        img_name = img_path.join(image_total[file])
        # Load the image file  从图形文件中读取图像，返回数组x*y*channal
        image = Image.open(img_path + image_total[file])
        image_str = np.double(np.array(image))
        #print(np.shape(image_str))
        #print(COLOR_TYPE)
        if operator.eq(COLOR_TYPE, "gray"):
            print('Making gray Image \n')
            if(np.equal(np.shape(image_str)[2], 3)):
                IMG = np.array(image.convert('L'))
                IMG = IMG / 255
            else:
                IMG = image_str
        elif operator.eq(COLOR_TYPE, "rgb"):
            print('Making rgb Image \n')
            IMG = image_str
            #  Normalize the RGB values to [0,1]
            IMG = IMG / 255

        # 原始图片维度Original image dimensions
        xdim = np.shape(IMG)[0]
        ydim = np.shape(IMG)[1]
        colors = np.shape(IMG)[2]
        orig_I.append(IMG)

        # Reshape the IMG
        if xdim >= ydim:
            maxdim = xdim
        else:
            maxdim = ydim
        PADIMG = np.zeros((maxdim, maxdim, colors))
        image_num = image_num + 1
        if image_num == 1:
            pad_I = np.zeros((maxdim, maxdim, colors, image_size))
        # 当图像x和y维度不一样时，要进行填充
        xpad = np.int((maxdim - xdim)/2)
        if (maxdim - xdim)%2 == 0:
           ypad = np.int((maxdim - xdim)/2)
        else:
           ypad = np.int((maxdim - xdim)/2) + 1
        for plane in range(colors):
            if(xdim >= ydim):
               PADIMG[:, :, plane] = np.lib.pad(IMG[:, :, plane], ((0, 0), (xpad, ypad)), 'constant', constant_values=0)
            elif(xdim < ydim):
               PADIMG[:, :, plane] = np.lib.pad(IMG[:, :, plane], ((xpad, ypad), (0, 0)), 'constant', constant_values=0)

        # 将填充的图片保存Store the padded images into a matrix
        #pad_I.append(PADIMG)
        pad_I[:, :, :, file] = PADIMG

        #print(xdim,ydim)
    # print(pad_I[1])
    #print(np.shape(pad_I[1]))
    print(np.shape(pad_I))  # [100,100,3,10]
    #print(orig_I[1])
    if(ZERO_MEAN):
        print("compute Mean")
        mn = np.mean(np.mean(pad_I, 0), 0) # [3,10]
        print(mn)
        sd = np.std(np.std(pad_I, 0), 0)
        for image in range(np.shape(pad_I)[3]):
            for i in range(np.shape(pad_I)[2]):
                orig_I[image][:, :, i] = orig_I[image][:, :, i] - mn[i, image]
                #orig_I[image][:, :, i] = orig_I[image][:, :, i] / sd[i, image]
    else:
        mn = []
        sd = []
        # print(orig_I[1])

    if (CONTRAST_NORMAILZE):
        #CN_I = np.zeros((maxdim, maxdim, colors, image_size))
        #res_I = []
        # 创建二维高斯核
        kernal = gaussian_2d_kernal(5, 1.5) # [5,5]
        print(kernal)
        print(np.shape(kernal))
        CN_I = []
        #global  new_dim
        new_dim = np.zeros((xdim-4, ydim-4, colors))
        I = np.zeros((xdim-4, ydim-4, colors, image_num))
        for image in range(np.shape(pad_I)[3]):
            print('Contrast Normalizing Image with Local CN: %d \n'% image)
            for j in range(np.shape(pad_I)[2]):
                dim = np.double(orig_I[image][:, :, j]) #[100,100]
                lmn = signal.convolve2d(dim, kernal, mode='valid')# [96,96]
                lmnsq = signal.convolve2d(dim ** 2, kernal, mode='valid') #[96,96]
                lvar = lmnsq - lmn ** 2
                lvar = np.maximum(lvar, 0) # 将小于0的元素置0
                lstd = np.sqrt(lvar)
                lstd = np.maximum(lstd, 1) # 将小于1的元素置1

                shifti = np.shape(kernal)[0] // 2 + 1
                shiftj = np.shape(kernal)[1] // 2 + 1

                # since we do valid convolutions
                dim = dim[shifti:shifti+np.shape(lstd)[0], shiftj:shiftj+np.shape(lstd)[1]]
                dim = dim - lmn

                dim = dim / lstd
                #print(np.shape(new_dim))
                #print(np.shape(dim))
                # for i in range(np.shape(dim)[0]):
                #     for j in range(np.shape(dim)[1]):
                #         dim[i, j] = np.double(dim[i, j]/lstd[i, j])
                if(image == 0 and j ==0):
                    if(np.shape(lmn)[0] >= np.shape(lmn)[1]):
                        maxdim = np.shape(lmn)[0]
                    else:
                        maxdim = np.shape(lmn)[1]
                new_dim[:, :, j] = dim
            CN_I.append(new_dim)
            I[:, :, :, image] = CN_I[image]
    else:
        CN_I = orig_I
    print(np.shape(CN_I[1])) # [96,96,3]

    # print(np.shape(I)) [96,96,3,10]

    return I, np.shape(I)[0], np.shape(I)[1], mn, sd





