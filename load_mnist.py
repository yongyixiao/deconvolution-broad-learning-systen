import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import operator
from scipy import signal
#
# mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#
# train_x, train_y = mnist.train.next_batch(3000)
# test_x = mnist.test.images[:1000]
# test_y = mnist.test.labels[:1000]
# #train_x = mnist.train.images[:]
# #train_y = mnist.train.labels[:]
# train = train_x.T
# a = np.reshape(train, (28, 28, 1, np.shape(train)[1]))



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

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def loadImages(CONTRAST_NORMAILZE, ZERO_MEAN, COLOR_TYPE, SQUARE_IMAGES):
    train_x, train_y = mnist.train.next_batch(200)
    train_x = train_x / 255
    train = train_x.T
    test_x = mnist.test.images[:50]
    test_y = mnist.test.labels[:50]
    test_x = test_x / 255
    test = test_x.T
    #train_x = mnist.train.images[:]
    #train_y = mnist.train.labels[:]
    all_images = np.reshape(train, (28, 28, 1, np.shape(train_x.T)[1])) # [28,28,1,100]
    test_images = np.reshape(test, (28, 28, 1, np.shape(test_x.T)[1]))
    #for i in range(np.shape(train_x)[0]):
    print("loading image")
    if operator.eq(COLOR_TYPE, 'gray'):
        print('Making gray Image \n')
    elif operator.eq(COLOR_TYPE, 'rgb'):
        print('Making rgb Image \n')

    if ZERO_MEAN:
        print("compute Mean")
        mn = np.mean(np.mean(all_images, 0), 0) # [1,100]
        test_mn = np.mean(np.mean(test_images, 0), 0)
        sd = np.std(np.std(all_images, 0), 0)
        print(np.shape(mn))
        for img_num in range(np.shape(all_images)[3]):
            for i in range(np.shape(all_images)[2]):
                all_images[:, :, i, img_num] = all_images[:, :, i, img_num] - mn[i, img_num]
        for img_num in range(np.shape(test_images)[3]):
            for i in range(np.shape(test_images)[2]):
                test_images[:, :, i, img_num] = test_images[:, :, i, img_num] - test_mn[i, img_num]
    else:
        mn = []
        sd = []

    xdim = np.shape(all_images)[0]
    ydim = np.shape(all_images)[1]
    colors = np.shape(all_images)[2]
    image_num = np.shape(all_images)[3]
    test_num = np.shape(test_images)[3]

    if CONTRAST_NORMAILZE:
        # 创建二维高斯核
        kernal = gaussian_2d_kernal(5, 1.5) # [5,5]
        print(kernal)
        new_dim = np.zeros((xdim - 4, ydim - 4, colors))
        I = np.zeros((xdim - 4, ydim - 4, colors, image_num))
        test_I = np.zeros((xdim - 4, ydim - 4, colors, test_num))
        for image in range(image_num):
            print('Contrast Normalizing Image with Local CN: %d \n'% image)
            for j in range(colors):
                dim = np.double(all_images[:, :, j, image]) # [28,28]
                lmn = signal.convolve2d(dim ** 2, kernal, mode='valid') # [24,24]
                lmnsq = signal.convolve2d(dim ** 2, kernal, mode='valid') #[24,24]
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
                new_dim[:, :, j] = dim
            I[:, :, :, image] = new_dim  # [24,24,1,100]

        for image in range(test_num):
            print('Contrast Normalizing Image with Local CN: %d \n' % image)
            for j in range(colors):
                dim = np.double(test_images[:, :, j, image]) # [28,28]
                lmn = signal.convolve2d(dim ** 2, kernal, mode='valid') # [24,24]
                lmnsq = signal.convolve2d(dim ** 2, kernal, mode='valid') #[24,24]
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
                new_dim[:, :, j] = dim
            test_I[:, :, :, image] = new_dim  # [24,24,1,100]

    return I, train_y, test_I, test_y, np.shape(I)[0], np.shape(I)[1], mn, sd

#I, train_y, xdim, ydim, mn, sd  = loadImages(1,1,'gray',1)
# print(np.shape(I))
#print(I[:,:,0,88])
