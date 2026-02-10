import numpy
from collections import OrderedDict
from layers import *

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
    
def SGD(net, x_batch, t_batch, iter_num, learning_rate):
    for i in range(iter_num):
        net.loss(x_batch, t_batch)
        d = net.lastLayer.backward(1.0)
        for s in ('ReLU2', 'Affine2', 'ReLU1', 'Affine1'):
            d = net.layers[s].backward(d)
        for s in ('Affine1', 'Affine2'):
            net.layers[s].w -= net.layers[s].dw * learning_rate
            net.layers[s].b -= net.layers[s].db * learning_rate
