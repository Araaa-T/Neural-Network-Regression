import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate = 0.01):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def weighted_sum(self, X):
        return np.dot(self.weights, X) + self.bias
    
    def fit(self, X, y_true, n_epoch=1000):
        for _ in range(n_epoch):
            for x_i, y_i in zip(X, y_true):
                y_pred = self.weighted_sum(x_i)
                error = y_i - y_pred
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
    def predict(self, X):
        if len(X) > 1:
            return np.array([self.weighted_sum(x) for x in X])
        return self.weighted_sum(X)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    k = np.random.randint(1, 10)
    b = np.random.randint(1, 10)
    data = np.random.rand(1000)
    # print(data)
    k, b = 125, 67

    y = k * data + b
    # print(y.shape)
    # print(f"Real y: {y}")
    error = (np.random.rand(y.shape[0]) - 9.5) / 10
    y_synthetic = y + error
    # print(y_synthetic)

    plt.plot(data, y_synthetic, 'o')
    plt.plot(data, y)
    plt.savefig('test.png')

X_train = data.reshape(-1, 1)
print(X_train)

model = Perceptron(1, learning_rate=0.1)
model.fit(data, y_synthetic, n_epoch=10)

y_pred = model.predict(data)

print(f"True k={k}, b={b}")
print(f"Learning Paraeters: w = {model.weights[0]}, b = {model.bias}")

plt.scatter(data, y_synthetic, label="Synthetic Data", color = "gray", alpha=0.5)
plt.plot(data, y, label="Real data", color = "blue", linewidth = 2)
plt.plot(data, y_pred, label="Model's prediction", color = "red", linestyle = "--")
plt.legend()
plt.savefig("test.png")

print(model.predict(np.array([354, 159])))s