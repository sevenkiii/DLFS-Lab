# Train and Test

import time
import matplotlib.pyplot as plt

from simpleNet import *
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label = True, normalize = True)

network = simpleNet(784, 100, 10)

loss_record = []
train_acc_record = []
test_acc_record = []


train_size = x_train.shape[0]
iter_num = 10000
batch_size = 100
gd_iter = 1
learning_rate = 0.21
epoch_per_iter = int(train_size / batch_size)
additional_info = " "

st_time = time.time()
if True:
    test_acc = network.accuracy(x_test, t_test)
    train_acc_record.append(test_acc)
    test_acc_record.append(test_acc)
for i in range(iter_num):
    idx = numpy.random.choice(train_size, batch_size)
    x_batch, t_batch = x_train[idx], t_train[idx]
    SGD(network, x_batch, t_batch, gd_iter, learning_rate)
    loss_record.append(network.loss(x_batch, t_batch))
    
    # print(f"Iter {i} Finished in {ela} seconds")
    if (i + 1) % epoch_per_iter == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_record.append(train_acc)
        test_acc_record.append(test_acc)
        ed_time = time.time()
        ela = ed_time - st_time
        print("Epoch " + str(int(i/epoch_per_iter)) + \
              ": Train acc = " + str(train_acc) + \
              " | Test acc = " + str(test_acc) + \
              " | Finished in " + str(ela) + " seconds")
        st_time = ed_time

x_axis = numpy.arange(0, iter_num)
plt.figure()
plt.plot(x_axis, loss_record)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("Training Loss " + additional_info)
plt.show()

x_axis = numpy.arange(0, len(test_acc_record))
plt.figure()
plt.plot(x_axis, train_acc_record, label="train acc")
plt.plot(x_axis, test_acc_record, linestyle="--", label="test acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Accuracy " + additional_info)
plt.legend()
plt.show()
