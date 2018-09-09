from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_x, train_y = mnist.train.next_batch(500)
a = 0
for i in range(100):
    for j in range(len(train_y[i])):
        if train_y[i][j] == 1:
            a = j
    print(a)