import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255. # normalize

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255. # normalize
_,m_train = X_train.shape

def __init__(input_num, neurons, output_num):
    w1 = np.random.rand(neurons, input_num) -0.5
    b1 = np.random.rand(neurons, 1)-0.5
    w2 = np.random.rand(neurons, output_num)-0.5
    b2 = np.random.rand(output_num, 1)-0.5
    return w1, b1, w2, b2


def tanh_funct(z):
    return np.tanh(z)


def tanh_deriv(z):
    a = tanh_funct(z)
    return 1 - a ** 2


def softmax_activation_function(z):
    return np.exp(z) / sum(np.exp(z))


def run_forward_propagation(weight1, bias1, weight2, bias2, X):
    z1 = weight1.dot(X) + bias1
    a1 = tanh_funct(z1)
    z2 = weight2.dot(a1) + bias2
    a2 = softmax_activation_function(z2)
    return z1, a1, z2, a2


def relu_derivative(z):
    return z > 0
#dz1 = w2.T.dot(dz2) * relu_derivative(z1)


def relu_activation_function(z):
    return np.maximum(z, 0)
    # a1 = relu_activation_function(z1)


def back_propagation(z1, a1, a2, w2, Y, X):
    Y_encoded = np.zeros((Y.size, Y.max() + 1))
    Y_encoded[np.arange(Y.size), Y] = 1
    Y_encoded = Y_encoded.T
    dz2 = a2 - Y_encoded
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * tanh_deriv(z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def update_weights_biases(w1, w2, b1, b2, dw1, dw2, db1, db2, rate):
    w1 = w1 - rate * dw1
    b1 = b1 - rate * db1
    w2 = w2 - rate * dw2
    b2 = b2 - rate * db2
    return w1, b1, w2, b2


def train(X, Y, alpha, epochs):
    weight1, bias1, weight2, bias2 = __init__(784, 10, 10)
    for i in range(epochs):
        z1, a1, z2, a2 = run_forward_propagation(weight1, bias1, weight2, bias2, X)
        dw1, db1, dw2, db2 = back_propagation(z1, a1, a2, weight2, Y, X)
        weight1, bias1, weight2, bias2 = update_weights_biases(weight1, weight2, bias1, bias2, dw1, dw2, db1, db2, alpha)
        if i % 10 == 0:
            print("epochs passed:", i)
            print(get_accuracy(a2, Y))
            print(" ")
    return weight1, bias1, weight2, bias2


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(a2, Y):
    predictions = get_predictions(a2)
    percentage = np.sum(predictions == Y) / Y.size
    return percentage

W1, B1, W2, B2 = train(X_train, Y_train, 0.1, 500)

#testing

def predict(X, w1, b1, w2, b2):
    _, _, _, a2 = run_forward_propagation(w1, b1, w2, b2, X)
    predictions = get_predictions(a2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = predict(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(235, W1, B1, W2, B2)
