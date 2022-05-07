import torch
import time 
from groupsort.groupsort_cuda import GroupSort as CudaGroupSort
from torch.autograd import grad

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.st_time = time.time()

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        
        print("{}: {:.8f}".format(self.label, time.time() - self.st_time))


def group_sort_cuda(x, groupsort, group_size):
    new_x = x.view(-1, group_size, f//group_size)
    return groupsort(new_x).view(b, f)

def group_sort_torch(x, group_size):
    return x.view(-1, group_size, f//group_size).sort(dim=1).values.view(b, f)

def group_sort_naive_3(x):
    a, b, c = x.split(f//3, dim=-1)

    max_e = torch.maximum(torch.maximum(a, b), c)
    min_e = torch.minimum(torch.minimum(a, b), c)
    mid_e = a + b + c - max_e - min_e
    return torch.cat([min_e, mid_e, max_e], dim=1)

def group_sort_naive_2(x):
    a, b = x.split(f//2, dim=-1)
    return torch.cat([torch.minimum(a, b), torch.maximum(a, b)], dim=1)

b = 5000
f = 2400


# forward pass testing
for group_size in range(1, 60):
    if f%group_size != 0: continue

    x = torch.randn(b, f).cuda()

    cuda_sort = CudaGroupSort(1, group_size=group_size)

    cuda_result = group_sort_cuda(x, cuda_sort, group_size=group_size)
    torch_result = group_sort_torch(x, group_size=group_size)

    st_time = time.time()
    for i in range(100):
        _ = group_sort_torch(x, group_size=group_size)
    ed_time = time.time()  
    torch_time = "torch: {:.8f}".format(ed_time - st_time)

    st_time = time.time()
    for i in range(100):
        _ = group_sort_cuda(x, cuda_sort, group_size=group_size)
    ed_time = time.time()  
    cuda_time = "cuda: {:.8f}".format(ed_time - st_time)

    # print(group_size, cuda_time)
    print(group_size, "equivalent:", torch_result.allclose(cuda_result), "TIME:", cuda_time, torch_time)

# backward pass testing
x = torch.randn((b, f), requires_grad=True).cuda()

for group_size in range(1, 60):
    if f%group_size != 0: continue

    view_x = x.view(-1, group_size, f//group_size)
    view_x.requires_grad_()

    cuda_sort = CudaGroupSort(1, group_size=group_size)
    cuda_output = cuda_sort(view_x)
    cuda_o = (cuda_output - view_x).abs().sum()
    cuda_grad = grad(cuda_o, view_x)[0]

    torch_output, indices = view_x.sort(dim=1, stable=True)
    torch_o = (torch_output - view_x).abs().sum()
    torch_grad = grad(torch_o, view_x)[0]
    # t_grad = torch_grad.gather(1, indices)


    print(group_size, "equivalent:", torch_grad.allclose(cuda_grad))
    

