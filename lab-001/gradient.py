import numpy
def numerical_gradient(f, x):
    h = 1e-4
    grad = numpy.zeros_like(x)
    it = numpy.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        x[idx] += h
        v = f(x)
        x[idx] -= 2*h
        v -= f(x)
        x[idx] += h
        grad[idx] = v / (2*h)
        it.iternext()
    return grad
