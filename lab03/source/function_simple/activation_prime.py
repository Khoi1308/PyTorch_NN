import torch


class ActivationPrime:
    # derivative of sigmoid
    @staticmethod
    def sigmoid_derivative(s):
        return (4 * torch.exp(2*s)) / ((1 + torch.exp(2*s)) * (1 + torch.exp(2*s)))

    # derivative of tanh
    @staticmethod
    def tanh_derivative(s):
        return 1 - torch.tanh(s) ** 2
