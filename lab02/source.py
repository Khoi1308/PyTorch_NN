import torch
import numpy as np
import torch.nn as nn

x_np = np.array([[1, 0, 2], [2, 0, 1]])
x_torch = torch.from_numpy(x_np)

# x_np += 1
x_torch += 1
print('x_np', x_np)
print("x_torch", x_torch)

y_torch = torch.tensor(([0, 8], [0, 4], [20, 20]),
dtype=torch.float)
y_np = y_torch.numpy()

# y_np += 1
y_torch +=1
print('y_torch', y_torch)
print('y_np', y_np)

# sigmoid activation
def sigmoid(s):
    return (torch.exp(s) - torch.exp(-s)) / (torch.exp(s) + torch.exp(-s))

# derivative of sigmoid
def sigmoid_derivative(s):
    return (4 * torch.exp(2*s)) / ((1 + torch.exp(2*s)) * (1 + torch.exp(2*s)))

# Feed Forward Neural Network class
class FFNN(nn.Module):
    # initialization function
    def __init__(self):
        # init function of base class
        super(FFNN, self).__init__()
        # corresponding size of each layer
        self.inputSize = 4
        self.hiddenSize = 4
        self.outputSize = 1
        # random weights from a normal distribution
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        # 4 X 4 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        # 4 X 1 tensor

        self.z = None
        self.z_activation = None
        self.z_activation_derivative = None
        self.z2 = None
        self.z3 = None
        self.out_error = None
        self.out_delta = None
        self.z2_error = None
        self.z2_delta = None

    # activation function using sigmoid
    def activation(self, z):
        self.z_activation = sigmoid(z)
        return self.z_activation

    # derivative of activation function
    def activation_derivative(self, z):
        self.z_activation_derivative = sigmoid_derivative(z)
        return self.z_activation_derivative   #

    # forward propagation
    def forward(self, X):
        # multiply input X and weights W1 from input layer to hidden layer
        self.z = torch.matmul(X, self.W1)
        self.z2 = self.activation(self.z) # activation function
        # multiply current tensor and weights W2 from hidden layer to output layer
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.activation(self.z3) # final activation function
        return o

    # backward propagation
    def backward(self, X, y, o, rate):
        self.out_error = y - o # error in output
        self.out_delta = self.out_error * self.activation_derivative(o) # derivative of activation to error
        # error and derivative of activation to error of next layer in backward propagation
        self.z2_error = torch.matmul(self.out_delta,
                                     torch.t(self.W2))
        self.z2_delta = self.z2_error * self.activation_derivative(self.z2)
        # update weights from delta of error and learning rate
        self.W1 += torch.matmul(torch.t(X), self.z2_delta) * rate
        self.W2 += torch.matmul(torch.t(self.z2), self.out_delta) * rate
    # backward propagation

    # training function with learning rate parameter
    def train(self, X, y, rate):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o, rate)

    # save weights of model
    @staticmethod
    def save_weights(model, path):
        # use the PyTorch internal storage functions
        torch.save(model, path)

    # load weights of model
    @staticmethod
    def load_weights(path):
    # reload model with all the weights
        torch.load(path)

    # predict function
    def predict(self, x_predict):
        print("Predicted data based on trained weights: ")
        print("Input: \n" + str(x_predict))
        print("Output: \n" + str(self.forward(x_predict)))


# sample input and output value for training
X = torch.tensor(([2, 9, 0, 6], [1, 5, 1, 1], [3, 6, 2, 4], [2, 1, 4, 5]), dtype=torch.float)  # 4 X 4 tensor
y = torch.tensor(([90], [100], [88], [70]), dtype=torch.float)  # 4 X 1 tensor

# scale units by max value
X_max, _ = torch.max(X, 0)
X = torch.div(X, X_max)
y = y / 100  # for max test score is 100

# sample input x for predicting
x_predict = torch.tensor(([3, 8, 4, 2]), dtype=torch.float)  # 1 X 4 tensor

# scale input x by max value
x_predict_max, _ = torch.max(x_predict, 0)
x_predict = torch.div(x_predict, x_predict_max)

# create new object of implemented class
NN = FFNN()

# trains the NN 1,000 times
for i in range(1000):
    # print mean sum squared loss
    print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X)) ** 2).detach().item()))
    # training with learning rate = 0.4
    NN.train(X, y, 0.4)
# save weights
NN.save_weights(NN, "NN")

# load saved weights
NN.load_weights("NN")
# predict x input
NN.predict(x_predict)
