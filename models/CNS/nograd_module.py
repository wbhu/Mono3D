import torch
import torch.nn as nn
from torch.autograd.function import Function


class NogradFunction(Function):
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input


class NogradModule(nn.Module):
    def __init__(self, nograd_fun):
        super(NogradModule, self).__init__()
        self.nograd_fun = nograd_fun

    def forward(self, input):
        output = self.nograd_fun(input)
        return output

    def backward(self, input):
        output = self.nograd_fun(input)
        return output


class QuantizeFun(NogradFunction):
    def __init__(self, lvl=255):
        super(QuantizeFun, self).__init__()
        self.lvl = lvl

    def forward(self, input):
        output = torch.clamp(input, min=0, max=1)
        output = torch.mul(output, self.lvl)
        output = torch.round(output)
        output = torch.mul(output, 1 / self.lvl)
        return output
