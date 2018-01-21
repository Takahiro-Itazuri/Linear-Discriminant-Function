import numpy as np
import matplotlib.pyplot as plt

def predict(w, x):
    return np.dot(w, x)

def activation_function(x):
    if x > 0:
        return 1
    else:
        return -1

# datasets
data = np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
label = np.array([1, -1, -1, -1])

# 正解データのグラフ描画
ax = plt.subplot(1, 2, 1)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("ground truth")

for i in range(len(data)):
    if label[i] == 1:
        plt.plot(data[i][0], data[i][1], "ro")
    else:
        plt.plot(data[i][0], data[i][1], "bo")


# 初期化
w = np.array(np.random.normal(loc=0.0, scale=1.0, size=3))

epochs = 100
learning_rate = 0.1
for e in range(epochs):
    for i in range(len(data)):
        plabel = activation_function(predict(w, data[i]))
        if plabel != label[i]:
            w += learning_rate * data[i] * label[i]

# 予測
ax = plt.subplot(1, 2, 2)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("result")

for i in range(len(data)):
    plabel = activation_function(predict(w, data[i]))
    
    if plabel == 1:
        plt.plot(data[i][0], data[i][1], "ro")
    else:
        plt.plot(data[i][0], data[i][1], "bo")

x = np.arange(-2, 2, 0.1)
y = (-1) / w[1] * (w[0] * x + w[2])
plt.plot(x, y, "g")

plt.savefig("LinearDiscriminantFunction")
    