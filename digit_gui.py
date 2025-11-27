import tkinter as tk
import numpy as np

class DigitGUI:
    def __init__(self, model, scale=10):
        self.model = model
        self.scale = scale
        self.size = 28 * scale
        self.buf = np.zeros((28, 28), dtype=np.float32)
        self.root = tk.Tk()
        self.root.title("MNIST CNN")
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.pred_var = tk.StringVar()
        self.pred_label = tk.Label(self.root, textvariable=self.pred_var, font=("Arial", 16))
        self.pred_label.grid(row=1, column=0, sticky="w")
        self.btn_predict = tk.Button(self.root, text="Predict", command=self.predict)
        self.btn_predict.grid(row=1, column=1)
        self.btn_clear = tk.Button(self.root, text="Clear", command=self.clear)
        self.btn_clear.grid(row=1, column=2)
        self.last = None
        self.canvas.bind('<ButtonPress-1>', self.on_down)
        self.canvas.bind('<B1-Motion>', self.on_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_up)

    def on_down(self, e):
        self.last = (e.x, e.y)
        self.stamp(e.x, e.y)

    def on_move(self, e):
        if self.last is not None:
            x0, y0 = self.last
            x1, y1 = e.x, e.y
            self.canvas.create_line(x0, y0, x1, y1, width=12, fill='black', capstyle=tk.ROUND, smooth=True)
            self.last = (x1, y1)
            self.stamp(x1, y1)

    def on_up(self, e):
        self.last = None

    def stamp(self, x, y):
        gx = max(0, min(27, int(x // self.scale)))
        gy = max(0, min(27, int(y // self.scale)))
        r = 2
        x0 = max(0, gx - r)
        x1 = min(28, gx + r + 1)
        y0 = max(0, gy - r)
        y1 = min(28, gy + r + 1)
        self.buf[y0:y1, x0:x1] = 1.0

    def softmax(self, x):
        a = x.max(axis=1, keepdims=True)
        e = np.exp(x - a)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self):
        x = self.buf.reshape(1, 1, 28, 28).astype(np.float32)
        logits = self.model.forward(x)
        probs = self.softmax(logits)
        cls = int(np.argmax(probs[0]))
        p = float(probs[0, cls])
        self.pred_var.set(f"Pred: {cls}  prob: {p:.3f}")

    def clear(self):
        self.canvas.delete('all')
        self.buf[:] = 0.0
        self.pred_var.set("")

    def run(self):
        self.root.mainloop()

