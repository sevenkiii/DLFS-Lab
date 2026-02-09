import numpy

class layerAffine:
    def __init__(self, input_size, output_size, weight_std=0.01):
        self.w = weight_std * numpy.random.randn(input_size, output_size)
        self.b = numpy.zeros(output_size)
    def forward(self, x):
        return numpy.dot(x, self.w) + self.b
    
class layerReLU:
    def forward(self, x):
        return numpy.maximum(x, 0)
    
class SoftmaxWithLoss:
    def forward(self, x, t):
        if x.ndim == 1:
            x.reshape(1, x.shape[0])
        batch_size = x.shape[0]
        maxValue = numpy.max(x, axis=1, keepdims=True)
        s = numpy.exp(x - maxValue) / numpy.sum(numpy.exp(x - maxValue), axis=1, keepdims=True)
        return numpy.sum(t * -numpy.log(s + 1e-7)) / batch_size

