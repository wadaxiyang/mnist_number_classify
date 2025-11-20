
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dY):
        return dY * self.mask
