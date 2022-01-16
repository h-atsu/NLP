import utils
import numpy as np
from datasets import ptb
import dezero.layers as L
import dezero.functions as F
from dezero import Variable

c = Variable(np.array([[1., 0., 0., 0.]]))
W = Variable(np.random.randn(4, 3))

v = F.matmul(c, W)
print(v)
