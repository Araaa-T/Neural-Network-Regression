import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
# print(X)
y_real = 4 + 3 * X
y_noisy = y_real + np.random.uniform(-0.3, 0.3, (1000, 1))
# print(y_real)
# print(y_noisy)

# plt.plot(X, y_real, label="Real data", color="blue")
# plt.scatter(X, y_noisy, label="Synthetic data", color="red")
# plt.legend()
# plt.savefig("test2.png")

class Perceptron:
    def __init__(self, lr = 0.01):
        self.weight = np.random.randn(1)
        self.bias = np.random.randn(1)
        self.lr = lr
    def predict(self, X):
        return X * self.weight + self.bias
    
    def train_step(self, X_batch, y_batch):
        y_pred = self.predict(X_batch)

        error = y_batch - y_pred
        dw = 2 * np.mean(error * X_batch)
        db = 2 * np.mean(error)

        self.bias += self.lr * db
        self.weight += self.lr * dw

        return np.mean(error ** 2)
    
model = Perceptron(lr=0.01)
batch_size = 16
epochs = 100
n_samples = len(X)
losses = []

for epoch in range(epochs):
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y_noisy[indices]

    epoch_loss = 0

    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        current_loss = model.train_step(X_batch, y_batch)
        epoch_loss += current_loss
    
    avg_loss = epoch_loss / (n_samples / batch_size)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {avg_loss}, w = {model.weight[0]}, b = {model.bias[0]}")


print(model.predict(0.3))
np_losses = np.array(losses)
print(f"Losses during epochs: \n {np_losses}")

# Matplotlib

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss reduction over epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.subplot(1, 2, 2)
plt.scatter(X, y_noisy, color='red', alpha=0.2, label="Noisy Data")
plt.plot(X, y_real, color='blue', linewidth=2, label="Original (w=3, b=4)")
plt.plot(X, model.predict(X), color='green', linestyle='--', label="Predicted")
plt.legend()

plt.tight_layout()
plt.savefig("result.png")

# y_real = 4 + 3 * X ==> w = 3 and b = 4
from mpl_toolkits.mplot3d import Axes3D

w_range = np.linspace(0, 3, 50)
b_range = np.linspace(0, 4, 50)
print(f"Possible w values: \n {w_range}")
print(f"Possible b values: \n {b_range}")
W, B = np.meshgrid(w_range, b_range)

def calculate_mse(w_val, b_wal, X, y):
    predictions = X * w_val + b_wal
    return np.mean((y - predictions) ** 2) 

Z = np.array([calculate_mse(w_i, b_i, X, y_noisy) for w_i, b_i in zip(np.ravel(W), np.ravel(B))])
Z = Z.reshape(W.shape)

print(Z)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.8, edgecolor='none')

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE Loss')
ax.set_title('MSE Loss Surface')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("loss_surface_3d.png")
# plt.show()