import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
"""
Scales all input data (as a lump) to roughly [-1,1] by shifting by the minimum up and
then dividing by the max value/2 and then making it zero mean. Since it makes
it zero mean there is no guarantee that the valeus will be [-1,1] after the scaling
"""
def rescale_all_n1_1(z):


    # minz = z.min()
    # z = z - minz
    # maxz = z.max()
    # z = z / (maxz * 2)
    # means = z.mean()
    # z = z - means
    new_z = np.reshape(z, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2]*np.shape(z)[3], 1))
    min_z = np.min(new_z)
    z = z - min_z
    new_z1 = np.reshape(z, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2]*np.shape(z)[3], 1))
    max_z = np.max(new_z1)
    z = z / max_z * 2
    new_z2 = np.reshape(z, (np.shape(z)[0]*np.shape(z)[1]*np.shape(z)[2]*np.shape(z)[3], 1))
    mean_z = np.mean(new_z2)
    z = z - mean_z

    return  z
    #print(z)

# rescale_all_n1_1(np.random.rand(3,3,3))

"""
在由边框分隔的指定颜色空间中显示一堆图像，并缩放为具有相同的对比度。
@:param imstack :the filters in xdim x ydim x color_planes x num_images
"""
def sdispims(imstack):

    if(imstack.ndim == 4):
        drows = np.shape(imstack)[0]
        dcols = np.shape(imstack)[1]
        numcolors = np.shape(imstack)[2]
        N = np.shape(imstack)[3]
    elif(imstack.ndim == 3): # If only one sample, it may appear as a 3D array
        drows = np.shape(imstack)[0]
        dcols = np.shape(imstack)[1]
        N = np.shape(imstack)[2]
        imstack = np.reshape(imstack, [drows, dcols, 3, 1])
        numcolors = 3
        N = 1
    elif(imstack.ndim == 2):
        drows = np.shape(imstack)[0]
        dcols = np.shape(imstack)[1]
        N = 1
        numcolors = 1
        imstack = np.reshape(imstack, [drows, dcols])

    fud = 0
    border = 2
    scaler = 1
    title = 'none'
    # 小数向上取整
    n2 = math.ceil(np.sqrt(N))
    COLOR_TYPES = 'rgb'

    # Size of each square.
    drb = drows + border  # 96 + 2
    dcb = dcols + border

    # Initialize the image size to -1 so that borders are black still
    imdisp = np.zeros((n2*drb-border, math.ceil(N/n2)*dcb-border, numcolors)) # [390.292,3]
    border_indices = np.ones((n2*drb-border, math.ceil(N/n2)*dcb-border, numcolors))

    A_image = []
    for i in range(N):
        A = np.double(imstack[:,:,:,i] - np.min(imstack[:]))
        maxA = np.max(imstack[:] - np.min(imstack[:]))
        A = A / maxA
        new_imstack = np.double(A * 255)
        A_image.append(new_imstack)
    # Scale the input values to be between zero and one
    # A = A_image - np.min(imstack[:])
    # maxA = np.max(imstack - np.min(imstack[:]))
    # A = A / maxA
    #new_imstack = (A * 255)


    plt.figure(figsize=(10, 5))
    plt.suptitle('Muti_Image')
    for nn in range(6):

        plt.subplot(3, 3, nn+1)
        plt.title('image' + str(1))
        plt.imshow(imstack[:, :, :, nn])
        plt.axis('on')
   # plt.imshow(A_image[1])
    plt.show()

    print(A_image[1])



def sdispmaps(imstack):


    return 1

# Normalizes each plane of the input(assumed to be in the first two dimensions
# of a 3 or 4-D matrix). This makes the each plane have unit length
def plane_normalize(value):

    xdim = np.shape(value)[0]
    ydim = np.shape(value)[1]

    # Columnize the planes [49,3,9]
    in_value = np.reshape(value, (xdim*ydim, np.shape(value)[2], np.shape(value)[3]))
    repeat_value = np.sqrt(np.sum(in_value**2,0))
    re_repeat_value = np.reshape(repeat_value, (1, np.shape(repeat_value)[0], np.shape(repeat_value)[1]))
    print(np.shape(re_repeat_value))
    remat_value = np.zeros((xdim*ydim, np.shape(value)[2], np.shape(value)[3]))
    for i in range(xdim*ydim):
        remat_value[i, :, :] = re_repeat_value
    #repeat_value = np.tile(re_repeat_value, (np.shape(in_value)[0], 1))
    print(np.shape(remat_value))
    in_value = in_value / repeat_value
    new_value = np.reshape(in_value, (xdim, ydim, np.shape(value)[2], np.shape(value)[3]))

    return new_value


# 生产随机序列
def generaterand(path, low, high):
    if low < high:
        mid = np.random.randint(low, high)
        path.append(mid)
        generaterand(path, low, mid-1)
        generaterand(path, mid+1, high)
    elif low == high:
        path.append(low)
    return path

"""
Solves the following component-wise separable problem
@:param v: target values v the size of the feautre maps.
@:param beta: the constant beta clamping to the feature maps.
@:param alpha: the sparsity norm.
@:param kappa: the coefficient on the sparsity term
@:return w:the best computed root per-pixel
"""

def solve_image(v, beta, alphavalue, kappa): #(z,10,0.8,1) z[102,102,9]
    rang = 10
    step = 0.0001
    a = []
    xx = np.arange(-rang, rang+step, step) # 一个等差数组
    #lookup_v = np.zeros((1, np.shape(temp)[0]))
    known_beta = []
    known_alpha = []
    temp = compute_w(xx, beta, alphavalue, kappa)
    lookup_v = temp
    vv = np.reshape(v, (np.shape(v)[0]*np.shape(v)[1]*np.shape(v)[2], 1))
    w = interp1d(xx.T, lookup_v[:].T, kind='linear', fill_value='extrapolate')(vv)
    w = np.reshape(w, (np.shape(v)[0], np.shape(v)[1], np.shape(v)[2]))
    # elif known_alpha[-1] == alpha:
    #     ind = np.shape(known_alpha)[0]
    #     #print(lookup_v[:])
    #     w = interp1d(xx, lookup_v[:], kind='linear')(v[:])
    #     w = np.reshape(w, (np.shape(v)[0], np.shape(v)[1], np.shape(v)[2]))
    return w

def compute_w(xx, beta, alphavalue, kappa): # [200001,]
    iterations = 4
    x = xx
    for a in range(iterations):
        fd = kappa * alphavalue * (np.sign(x) * (np.abs(x) ** (alphavalue-1))) + beta * (x-xx)
        fdd = kappa * alphavalue * (alphavalue-1) * (np.abs(x) ** (alphavalue-2)) + beta
        value = np.zeros((1, np.shape(x)[0]))
        for i in range(np.shape(x)[0]):
            if np.isnan(fd[i]):
                fd[i] = 0
            # if np.inf(fdd[i]):
            #     fdd[i] = 0
            if fdd[i] == 0:
                value[0][i] = 0
            else:
                value[0][i] = fd[i] / fdd[i]
            x[i] = x[i] - value[0][i]
    # check whether the zero solution is the better one
    zz = (beta / 2) * (xx**2)
    f = kappa * (np.abs(x)**alphavalue) + (beta / 2) * ((x-xx)**2)
    for i in range(np.shape(f)[0]):
        if f[i] <= zz[i]:
            f[i] = 1
        else:
            f[i] = 0
    w = f * x
    return w
