import sys, os
import numpy as np
import matplotlib.pyplot as plt
from script.data_prepare import load_mnist
from script.network import FeedForwardNeuralNetwork
sys.path.append(os.pardir)  # 导入父目录


# 参数设定
iters_num = 30000  # 适当设定循环的次数
train_size = 50000  # x_train.shape[0]
batch_size = 100
learning_rate = 0.2
learning_threshold = 0.98
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 将神经网络的类实例化
my_network = FeedForwardNeuralNetwork(input_size=784, hidden_1_size=30, hidden_2_size=20, output_size=10)

# 训练数据并在测试集上做测试
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = my_network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        my_network.params[key] -= learning_rate * grad[key]

    # 计算损失函数
    loss = my_network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 统计准确度
    if i % iter_per_epoch == 0:
        train_acc = my_network.accuracy(x_train, t_train)
        test_acc = my_network.accuracy(x_test, t_test)
        if train_acc > learning_threshold:
            learning_rate = learning_rate/2

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形并保存结果
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.grid()
plt.savefig('../simout/accuracy_simout.png', dpi=400)
plt.show()
