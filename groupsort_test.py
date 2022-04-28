import torch
import time 
from maxmin.maxmin_cuda import MaxMin as CudaMaxMin
from maxmin.maxmin_py import MaxMin as PyMaxMin

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.st_time = time.time()

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        
        print("{}: {:.8f}".format(self.label, time.time() - self.st_time))


b = 5000
f = 2400

x = torch.randn(b, f).cuda()

def group_sort_cuda(x, group_size):
    maxmin = CudaMaxMin(1, group_size)
    new_x = x.view(-1, group_size, f//group_size)
    return maxmin(new_x).view(b, f)

def group_sort_torch(x, group_size):
    return x.view(-1, group_size, f//group_size).sort(dim=1, descending=True).values.view(b, f)

def group_sort_naive_3(x):
    a, b, c = x.split(f//3, dim=-1)

    max_e = torch.maximum(torch.maximum(a, b), c)
    min_e = torch.minimum(torch.minimum(a, b), c)
    mid_e = a + b + c - max_e - min_e
    return torch.cat([max_e, min_e, mid_e], dim=1)

def group_sort_naive_2(x):
    a, b = x.split(f//2, dim=-1)
    return torch.cat([torch.maximum(a, b), torch.minimum(a, b)], dim=1)

def group_sort_2(x):
    return x.view(-1, 2, f//2).sort(dim=1, descending=True).values.view(b, f)


# st_time = time.time()
# for i in range(500):
#     pymaxmin = group_sort_2_maxmin(x)
# ed_time = time.time()
# print("maxmin g=2: {:.8f}".format(ed_time - st_time))


# groupsize=2, naive
st_time = time.time()
for i in range(500):
  naive_2 = group_sort_naive_2(x)
ed_time = time.time()  
print("naive g=2: {:.8f}".format(ed_time - st_time))

# groupsize=2, pytorch
st_time = time.time()
for i in range(500):
  torch_2 = group_sort_torch(x, 2)
ed_time = time.time()
print("pytorch g=2: {:.8f}".format(ed_time - st_time))

# groupsize=2, cuda
st_time = time.time()
for i in range(500):
    cuda_2 = group_sort_cuda(x, 2)
ed_time = time.time()
print("cuda maxmin g=2: {:.8f}".format(ed_time - st_time))

print("correctness gs=2:")
print(torch.all(torch.eq(cuda_2, naive_2)))
print(torch.all(torch.eq(cuda_2, torch_2)))

# groupsize=3, naive
st_time = time.time()
for i in range(500):
  naive_3 = group_sort_naive_3(x)
ed_time = time.time()
print("naive g=3: {:.8f}".format(ed_time - st_time))

# groupsize=3, torch
st_time = time.time()
for i in range(500):
  torch_3 = group_sort_torch(x, 3)
ed_time = time.time()
print("pytorch g=3: {:.8f}".format(ed_time - st_time))

# groupsize=3, cuda
st_time = time.time()
for i in range(500):
    cuda_3 = group_sort_cuda(x, 3)
ed_time = time.time()
print("cuda maxmin g=3: {:.8f}".format(ed_time - st_time))

print("correctness gs=3:")
print(torch.all(torch.eq(cuda_3, naive_3)))
print(torch.all(torch.eq(cuda_3, torch_3)))





