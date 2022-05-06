import torch
import time 
from maxmin.maxmin_cuda import MaxMin as CudaMaxMin
from maxmin.maxmin_py import MaxMin as PyMaxMin
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


def group_sort_cuda(x, maxmin, group_size):
    new_x = x.view(-1, group_size, f//group_size)
    return maxmin(new_x).view(b, f)

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
x = torch.randn(b, f).cuda()
# for group_size in range(1, 60):
#     if f%group_size != 0: continue

#     cuda_sort = CudaMaxMin(1, group_size=group_size)

#     cuda_result = group_sort_cuda(x, cuda_sort, group_size=group_size)
#     torch_result = group_sort_torch(x, group_size=group_size)

    # st_time = time.time()
    # for i in range(500):
    #     _ = group_sort_torch(x, group_size=group_size)
    # ed_time = time.time()  
    # torch_time = "torch: {:.8f}".format(ed_time - st_time)

    # st_time = time.time()
    # for i in range(500):
    #     _ = group_sort_cuda(x, cuda_sort, group_size=group_size)
    # ed_time = time.time()  
    # cuda_time = "cuda: {:.8f}".format(ed_time - st_time)

    # print(group_size, cuda_time)
    # print(group_size, "equivalent:", torch_result.allclose(cuda_result), "TIME:", cuda_time, torch_time)

# backward pass testing
group_size = 5
x = torch.randn((b, f), requires_grad=True).cuda()
view_x = x.view(-1, group_size, f//group_size)
view_x.requires_grad_()

cuda_sort = CudaMaxMin(1, group_size=group_size)
cuda_output = cuda_sort(view_x)
cuda_o = (cuda_output - view_x).abs().sum()
cuda_grad = grad(cuda_o, view_x)[0]

torch_output, indices = view_x.sort(dim=1)
torch_o = (torch_output - view_x).abs().sum()
torch_grad = grad(torch_o, view_x)[0]
# t_grad = torch_grad.gather(1, indices)

print("equivalent:", torch_grad.allclose(cuda_grad))



# st_time = time.time()
# for i in range(500):
#     pymaxmin = group_sort_2_maxmin(x)
# ed_time = time.time()
# print("maxmin g=2: {:.8f}".format(ed_time - st_time))


# # groupsize=2, naive
# st_time = time.time()
# for i in range(500):
#   naive_2 = group_sort_naive_2(x)
# ed_time = time.time()  
# print("naive g=2: {:.8f}".format(ed_time - st_time))

# # groupsize=2, pytorch
# st_time = time.time()
# for i in range(500):
#   torch_2 = group_sort_torch(x, 2)
# ed_time = time.time()
# print("pytorch g=2: {:.8f}".format(ed_time - st_time))


# cuda_sort = CudaMaxMin(1, group_size=2)
# # # groupsize=2, cuda
# # st_time = time.time()
# # for i in range(500):
# #     cuda_2 = group_sort_cuda(x, cuda_sort, 2)
# # ed_time = time.time()
# # print("cuda maxmin g=2: {:.8f}".format(ed_time - st_time))

# print("correctness gs=2:")
# cuda_2 = group_sort_cuda(x, cuda_sort, 2)
# naive_2 = group_sort_naive_2(x)
# torch_2 = group_sort_torch(x, 2)
# print(naive_2.allclose(cuda_2))
# print(torch_2.allclose(cuda_2))
# print(naive_2.allclose(torch_2))


# maxmin3 = CudaMaxMin(1, group_size=3)
# # groupsize=3, cuda
# st_time = time.time()
# for i in range(500):
#     cuda_3 = group_sort_cuda(x, maxmin3, 3)
# ed_time = time.time()
# print("cuda maxmin g=3: {:.8f}".format(ed_time - st_time))

# # groupsize=3, naive
# st_time = time.time()
# for i in range(500):
#   naive_3 = group_sort_naive_3(x)
# ed_time = time.time()
# print("naive g=3: {:.8f}".format(ed_time - st_time))

# # groupsize=3, torch
# st_time = time.time()
# for i in range(500):
#   torch_3 = group_sort_torch(x, 3)
# ed_time = time.time()
# print("pytorch g=3: {:.8f}".format(ed_time - st_time))

# cuda_3 = group_sort_cuda(x, maxmin3, 3)
# naive_3 = group_sort_naive_3(x)
# torch_3 = group_sort_torch(x, 3)

# print("correctness gs=3:")
# print(naive_3.allclose(cuda_3))
# print(torch_3.allclose(cuda_3))
# print(naive_3.allclose(torch_3))





