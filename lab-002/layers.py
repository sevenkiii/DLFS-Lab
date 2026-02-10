import numpy

class layerAffine:
    def __init__(self, input_size, output_size, weight_std=0.01):
        self.w = weight_std * numpy.random.randn(input_size, output_size)
        self.b = numpy.zeros(output_size)
        self.dw = None
        self.db = None
        self.x = None
    def forward(self, x):
        self.x = x
        return numpy.dot(x, self.w) + self.b
    def backward(self, dy):
        self.db = numpy.sum(dy, axis=0)
        self.dw = numpy.dot(self.x.T, dy)
        return numpy.dot(dy, self.w.T)

    
class layerReLU:
    def __init__(self):
        self.x = None
    def forward(self, x):
        self.x = x
        return numpy.maximum(x, 0)
    def backward(self, dy):
        return (self.x > 0) * dy
    
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        if x.ndim == 1:
            x.reshape(1, x.shape[0])
        batch_size = t.shape[0]
        maxValue = numpy.max(x, axis=1, keepdims=True)
        s = numpy.exp(x - maxValue) / numpy.sum(numpy.exp(x - maxValue), axis=1, keepdims=True)
        self.y = s
        return numpy.sum(t * -numpy.log(s + 1e-7)) / batch_size
    def backward(self, dL):
        batch_size = self.t.shape[0]
        return (self.y - self.t) * dL / batch_size

