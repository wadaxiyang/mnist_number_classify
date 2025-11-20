from myCNN import Conv2D, ReLU, MaxPool2D, Flatten, Linear

class CNN:
    def __init__(self):
        self.conv = Conv2D(1, 8, kernel_size=5, stride=1, pad=2, seed=1)
        self.relu = ReLU()
        self.pool = MaxPool2D(2, 2)
        self.flat = Flatten()
        self.fc = Linear(8 * 14 * 14, 10, seed=1)

    def forward(self, X):
        x = self.conv.forward(X)
        x = self.relu.forward(x)
        x = self.pool.forward(x)
        x = self.flat.forward(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dlogits):
        d, dW_fc, db_fc = self.fc.backward(dlogits)
        d = self.flat.backward(d)
        d = self.pool.backward(d)
        d = self.relu.backward(d)
        dX, dW_conv, db_conv = self.conv.backward(d)
        return (dW_conv, db_conv, dW_fc, db_fc)

    def step(self, grads, lr):
        dW_conv, db_conv, dW_fc, db_fc = grads
        self.conv.W -= lr * dW_conv
        self.conv.b -= lr * db_conv
        self.fc.W -= lr * dW_fc
        self.fc.b -= lr * db_fc

