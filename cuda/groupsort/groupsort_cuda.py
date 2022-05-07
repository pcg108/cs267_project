import torch
from groupsort import groupsort_extension

class GroupSortFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, axis, group_size):
        ctx.save_for_backward(input)
        ctx.axis = axis
        ctx.group_size = group_size
        output, argsort = groupsort_extension.forward(input, axis, group_size)
        ctx.argsort = argsort
        return output

    @staticmethod
    def backward(ctx, grad_h):
        input, = ctx.saved_tensors
        return groupsort_extension.backward(input, grad_h.contiguous(), ctx.axis, ctx.group_size, ctx.argsort), None, None

class GroupSort(torch.nn.Module):
    def __init__(self, axis=-1, group_size=2):
        super(GroupSort, self).__init__()
        self.axis = axis
        self.group_size = group_size

    def forward(self, x):
       return GroupSortFunction.apply(x, self.axis, self.group_size)
