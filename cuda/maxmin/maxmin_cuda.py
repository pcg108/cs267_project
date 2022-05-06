import torch
from maxmin import maxmin_extension

class MaxMinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, axis, group_size):
        ctx.save_for_backward(input)
        ctx.axis = axis
        ctx.group_size = group_size
        output, argsort = maxmin_extension.forward(input, axis, group_size)
        ctx.argsort = argsort
        return output

    @staticmethod
    def backward(ctx, grad_h):
        input, = ctx.saved_tensors
        return maxmin_extension.backward(input, grad_h.contiguous(), ctx.axis, ctx.group_size, ctx.argsort), None, None

class MaxMin(torch.nn.Module):
    def __init__(self, axis=-1, group_size=2):
        super(MaxMin, self).__init__()
        self.axis = axis
        self.group_size = group_size

    def forward(self, x):
       return MaxMinFunction.apply(x, self.axis, self.group_size)
