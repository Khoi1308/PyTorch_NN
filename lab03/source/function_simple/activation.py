import torch


class Activation:
    # sigmoid activation
    @staticmethod
    def sigmoid(s):
        return (torch.exp(s) - torch.exp(-s)) / (torch.exp(s) + torch.exp(-s))

    # tanh activation
    @staticmethod
    def tanh(s):
        return torch.tanh(s)
