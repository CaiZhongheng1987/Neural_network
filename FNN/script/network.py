from script.common_function import *
from collections import OrderedDict


# 将神经网络设计为一个类
class FeedForwardNeuralNetwork:
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size, weight_init_std=0.01):
        # 初始化神经网络的参数

        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_1_size)
        self.params['b1'] = np.zeros(hidden_1_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_1_size, hidden_2_size)
        self.params['b2'] = np.zeros(hidden_2_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_2_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # 神经网络的预测函数，主结构，调用每个单元的forward结构组合而成
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # 损失函数
        # x:输入数据, t:监督数据，就是图像标签
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # 计算准确度
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        # dout = 1  # 初始化
        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = dict()
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads


class Sigmoid:
    # Sigmoid的正向和负向传播函数
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    # W·X+b的正向和负向传播函数
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    # Softmax和Loss联合的正向和负向传播函数
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
