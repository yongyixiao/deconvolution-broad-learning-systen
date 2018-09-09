
from bls_train import *



# one shot solution using pseudo-inverse
s = 0.8   #增强节点的收缩比例
C = 2e-30  # L2正则化参数
N11 = 10  # feature nodes per window
N2 = 6 # number of windows of feature nodes
N33 = 150 # number of enhancement nodes
epochs = 1 # number of epochs
train_err = np.zeros((1,epochs))
test_err = np.zeros((1,epochs))
train_time = np.zeros((1,epochs))
test_time = np.zeros((1,epochs))

def bls_no(train_x, train_y, test_x, test_y):
    N1 = N11
    N3 = N33
    j = 1
    for j in range(epochs):
        train_accuracy, test_accuracy = bls_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3)

    return train_accuracy, test_accuracy




