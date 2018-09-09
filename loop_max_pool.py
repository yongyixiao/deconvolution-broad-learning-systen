import numpy as np
import math
import operator

def max_min_index(num, flag):
    index = []
    aa = []
    if operator.eq(flag, 'max'):
        maxA = np.max(num, 0)
        aa = maxA
        for i in range(np.shape(num)[1]):
            for j in range(np.shape(num)[0]):
                if maxA[i] == num[j][i]:
                    index.append(j)
    elif operator.eq(flag, 'min'):
        minA = np.min(num, 0)
        aa = minA
        for i in range(np.shape(num)[1]):
            for j in range(np.shape(num)[0]):
                if minA[i] == num[j][i]:
                    index.append(j)

    return aa, index



def loop_max_pool(input_img, pool_size):

    # [30,30,3]
    if len(np.shape(input_img)) == 3:
        xdim, ydim, numplanes = np.shape(input_img)
        numcases = 1
    else:
        xdim, ydim, numplanes, numcases = np.shape(input_img)

    # The pooled input planes (not dimensions 3 and 4 are reversed). [30,30,3]
    output = np.zeros((xdim, ydim, numplanes*numcases))
    new_output = np.reshape(output, (xdim*ydim*numplanes*numcases, 1))

    rblocks = math.ceil(xdim / pool_size[0]) # 15
    cblocks = math.ceil(ydim / pool_size[1]) # 15
    blockel = pool_size[0] * pool_size[1]  # 4 number of elements in block
    x = np.zeros((blockel, numcases*numplanes)) # [4,3]this is made double to work with randbinom

    # Loop over each plane
    for ii in range(rblocks):
        for jj in range(cblocks):
            x[:] = np.reshape(input_img[ii*pool_size[0] + 0 : ii*pool_size[0] + 2, jj*pool_size[0] + 0 : jj*pool_size[0] + 2, :],
                              (blockel, numplanes*numcases))
            maxA, maxind = max_min_index(x, 'max') # [1,3]
            minA, inds = max_min_index(x, 'min')
            maxes = minA # Iitialize to the mins (and their indices).
            for i in range(len(maxA)):
                if maxA[i] >= np.abs(minA[i]):
                    maxes[i] = maxA[i]
                    inds[i] = maxind[i]

            #  Compute offsets into the output planes.
            xoffset = []
            yoffset = []
            for i in range(len(maxes)):
                xoffset.append(((inds[i]-1) % pool_size[0] + 1))
                yoffset.append(((inds[i]-xoffset[i])/pool_size[0]+1+jj*pool_size[1]))
                xoffset[i] = xoffset[i] + ii * pool_size[0]
                a = i * xdim * ydim + (yoffset[i]-1) * xdim + xoffset[i]
                a = int(a)
                new_output[a][0] = maxes[i]

    output = np.reshape(new_output, (xdim, ydim, numcases*numplanes))

    return output


def max_pool(input_img, pool_size):
    # 24,24,1,100
    test_input = input_img
    if len(np.shape(input_img)) == 4:
       xdim, ydim, numplanes, numcases = np.shape(input_img)
    else:
       xdim, ydim, numplanes = np.shape(input_img)
       numcases = 1

    # The pooled input planes [12,12,100]
    pooled = np.zeros((math.ceil(xdim / pool_size[0]), math.ceil(ydim / pool_size[1]), numcases * numplanes))
    # Store the indices for each plane.
    indices = np.zeros((math.ceil(xdim / pool_size[0]), math.ceil(ydim / pool_size[1]), numplanes * numcases))
    rblocks = math.ceil(xdim / pool_size[0]) # 12
    cblocks = math.ceil(ydim / pool_size[1]) # 12
    blockel = pool_size[0] * pool_size[1]
    x = np.zeros((blockel, numcases * numplanes))

    input_img = np.reshape(input_img, (xdim, ydim, numplanes * numcases))

    for ii in range(rblocks):
        for jj in range(cblocks):
            x[:] = np.reshape(input_img[ii*pool_size[0] + 0 : ii*pool_size[0] + 2, jj*pool_size[0] + 0 : jj*pool_size[0] + 2, :],
                              (blockel, numplanes*numcases))
            maxA, maxind = max_min_index(x, 'max') # [1,100]
            minA, inds = max_min_index(x, 'min')
            maxes = minA
            for i in range(len(maxA)):
                if maxA[i] >= np.abs(minA[i]):
                    maxes[i] = maxA[i]
                    inds[i] = maxind[i]
            for k in range(numcases * numplanes):
                indices[ii, jj, k] = inds[k]
                pooled[ii, jj, k] = maxes[k]

    indices = np.reshape(indices, (np.shape(indices)[0], np.shape(indices)[1], numplanes, numcases))
    pooled = np.reshape(pooled, (np.shape(pooled)[0], np.shape(pooled)[1], numplanes, numcases))
    if len(np.shape(test_input)) == 3:
        pooled = np.reshape(pooled, (np.shape(pooled)[0], np.shape(pooled)[1], numplanes))
        indices = np.reshape(indices,(np.shape(indices)[0], np.shape(indices)[1], numplanes))

    return pooled, indices

# Undoes the max pooling by placing the max back into it's indexed location.
def reverse_max_pool(input_data, indices, pool_size, unpooled_size):

    if len(np.shape(input_data)) == 3:
        xdim, ydim, numplanes = np.shape(input_data)
        numcases = 1
    else:
        xdim, ydim, numplanes, numcases = np.shape(input_data)

    if len(np.shape(input_data)) == 3:
        indxdim, indydim, indplanes = np.shape(indices)
        indcases = 1
    else:
        indxdim, indydim, indplanes, indcases = np.shape(indices)
    # [24,24,1]
    unpooled = np.zeros((math.ceil(xdim * pool_size[0]), math.ceil(ydim * pool_size[1]), numplanes * numcases))
    new_unpooled = np.reshape(unpooled, (math.ceil(xdim * pool_size[0]) * math.ceil(xdim * pool_size[0]) * numplanes * numcases, 1))

    rblocks = xdim
    cblocks = ydim

    # For each sample there is an index so just use that to place max.
    # Make the indices into the rows.
    indices = np.reshape(indices, (np.shape(indices)[0], np.shape(indices)[1], indcases * indplanes))
    input_data = np.reshape(input_data, (xdim, ydim, numplanes))

    # Get blocks of the image

    for ii in range(rblocks):
        for jj in range(cblocks):
            inds = []
            ind_input = []
            ind = np.squeeze(indices[ii, jj, :])
            for i in range(indcases * indplanes):
                if indcases * indplanes == 1:
                    inds.append(ind)
                else:
                    inds.append(ind[i])
            input_ind = np.squeeze(input_data[ii, jj, :])
            for k in range(np.shape(input_data)[2]):
                if np.shape(input_data)[2] == 1:
                    ind_input.append(input_ind)
                else:
                    ind_input.append(input_ind[k])

            # Get offsets into the output image.
            xoffset = []
            yoffset = []
            for j in range(len(inds)):
                xoffset.append(((inds[j] - 1) % pool_size[0] + 1))
                yoffset.append(((inds[j] - xoffset[j]) / pool_size[0] + 1 + jj * pool_size[1]))
                xoffset[j] = xoffset[j] + ii * pool_size[0]
                a = j * np.shape(unpooled)[0] * np.shape(unpooled)[1] + (yoffset[j] - 1) * np.shape(unpooled)[0] + xoffset[j]
                a = int(a)
                new_unpooled[a][0] = ind_input[j]
    unpooled = np.reshape(new_unpooled, (np.shape(unpooled)[0], np.shape(unpooled)[1], numplanes))
    unpooled = unpooled[0:unpooled_size[0], 0:unpooled_size[1], :]

    return unpooled

# a = np.random.rand(12, 12, 1)
# #c= len(np.shape(a))
# b = np.ones((12,12,1))
# c = reverse_max_pool(a, b, [2,2],[24,24])
# print(c)