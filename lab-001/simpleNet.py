import numpy
from collections import OrderedDict
from layers import *
from gradient import numerical_gradient

class simpleNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = OrderedDict()
        self.layers['Affine1'] = layerAffine(input_size, hidden_size)
        self.layers['ReLU1'] = layerReLU()
        self.layers['Affine2'] = layerAffine(hidden_size, output_size)
        self.layers['ReLU2'] = layerReLU()
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = self.layers[layer].forward(x)
        return x
    
    def loss(self, x, t):
        x = self.predict(x)
        return self.lastLayer.forward(x, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        idx = numpy.argmax(y, axis=1)
        spv = numpy.argmax(t, axis=1)
        return numpy.sum(idx == spv) / spv.size

    def numerical_gradient(self, x, t):
        loss_func = lambda arg: self.loss(x, t) # Fake Function
        grad = {}
        grad['Affine1_w'] = numerical_gradient(loss_func, self.layers['Affine1'].w)
        grad['Affine1_b'] = numerical_gradient(loss_func, self.layers['Affine1'].b)
        grad['Affine2_w'] = numerical_gradient(loss_func, self.layers['Affine2'].w)
        grad['Affine2_b'] = numerical_gradient(loss_func, self.layers['Affine2'].b)
        return grad
    
def SGD(net, x_batch, t_batch, iter_num, learning_rate):
    for i in range(iter_num):
        grad = net.numerical_gradient(x_batch, t_batch)
        net.layers['Affine1'].w -= grad['Affine1_w'] * learning_rate
        net.layers['Affine1'].b -= grad['Affine1_b'] * learning_rate
        net.layers['Affine2'].w -= grad['Affine2_w'] * learning_rate
        net.layers['Affine2'].b -= grad['Affine2_b'] * learning_rate
