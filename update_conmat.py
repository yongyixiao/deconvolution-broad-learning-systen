from set_Defaults import *
import numpy as np
import random
import itertools
import operator
"""
Creates a connectivity matrix where the features maps connect to every input maps.
"""

"""
Creates a connectivity matrix where the features maps connect to every input maps.
@ param xdim This is first dimension of the connectivity matrix. This represents the number of input maps
@ param ydim This is the second dimension fo the connectivity matrix. This represents the number of features maps.
"""
def conmat_full(xdim,ydim):
    # Fully connected connectivity matrix is just ones [3,9]
    C = np.ones((xdim,ydim))
    #  Return the ydim for teh recommended size
    recommended = ydim
    return C, recommended

"""
Creates a connectivity matrix where the features maps connect to every
single input map and all possible pairs of input maps
"""
def conmat_singles_alldoub(xdim,ydim): # [9,45]
    C = np.zeros((xdim, ydim))
    for j in range(xdim):
        i = (j-1) % xdim + 1
        C[j, j] = 1
    # printj(C)
    xsize = np.int(xdim * (xdim-1) / 2)
    # 随机组合
    indices = list(itertools.combinations(range(1, xdim+1), 2))
    #  Put a one for each possible pair
    for j in range(np.shape(indices)[0]):
        C[indices[j][0]-1, j+xdim] = 1
        C[indices[j][1]-1, j+xdim] = 1
    recommended = xdim + xsize
    #print(C,recommended)
    return C, recommended

#conmat_singles_alldoub(9, 45)
"""
Creates a connectivity matrix where the features maps connect to randomly selected pairs of input maps.
"""
def conmat_randdoub(xdim, ydim):
    C = np.zeros(xdim, ydim)
    for j in range(ydim):
        li = range(xdim)
        indices = random.shuffle(li)
        C[indices[0], 1] = 1
        C[indices[1], 1] = 1
    recommended = ydim
    return C, recommended



def update_conmat(numlayer, conmat_types, num_input_maps, num_feature_maps):
    conmat = []
    new_num_feature_maps = []
    for layer in range(numlayer):

        if(operator.eq(conmat_types[layer], 'Full')):
            print("执行full")
            temp_conmat, recommended = conmat_full(num_input_maps[layer], num_feature_maps[layer])
            #print(temp_conmat)
        elif(operator.eq(conmat_types[layer], 'SAD')):
            print("执行sad")
            temp_conmat, recommended = conmat_singles_alldoub(num_input_maps[layer], num_feature_maps[layer])
            #print(temp_conmat)
        elif(operator.eq(conmat_types[layer], 'Random Doubles')):
            temp_conmat, recommended = conmat_randdoub(num_input_maps[layer], num_feature_maps[layer])
        conmat.append(temp_conmat)
        new_num_feature_maps.append(recommended)
    return conmat

# import itertools
# a = list(itertools.combinations(range(1,10),2))
# print(len((a)))
# print(a)






