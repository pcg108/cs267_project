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

def group_sort_2_maxmin(x):
    new_x = x.view(-1, 2, f//2)
    return maxmin(new_x)

def group_sort_2_maxmin_cuda(x):
    new_x = x.view(-1, 2, f//2)
    return maxmin(new_x)

def group_sort_3_sp(x):
    a, b, c = x.split(f//3, dim=-1)

    max_e = torch.maximum(torch.maximum(a, b), c)
    min_e = torch.minimum(torch.minimum(a, b), c)
    mid_e = a + b + c - max_e - min_e
    return torch.cat([max_e, min_e, mid_e], dim=1)

def group_sort_3(x):
    return x.view(-1, 3, f//3).sort(dim=1)

def group_sort_2_sp(x):
    a, b = x.split(f//2, dim=-1)
    return torch.cat([torch.maximum(a, b), torch.minimum(a, b)], dim=1)

def group_sort_2(x):
    return x.view(-1, 2, f//2).sort(dim=1)

maxmin = PyMaxMin(1)
st_time = time.time()
for i in range(500):
    y = group_sort_2_maxmin(x)
ed_time = time.time()
print("maxmin g=2: {:.8f}".format(ed_time - st_time))

maxmin = CudaMaxMin(1)
st_time = time.time()
for i in range(500):
    y = group_sort_2_maxmin_cuda(x)
ed_time = time.time()
print("cuda maxmin g=2: {:.8f}".format(ed_time - st_time))

st_time = time.time()
for i in range(500):
  y = group_sort_2_sp(x)

ed_time = time.time()
print("naive g=2: {:.8f}".format(ed_time - st_time))

st_time = time.time()
for i in range(500):
  y = group_sort_2(x)
ed_time = time.time()
print("pytorch g=2: {:.8f}".format(ed_time - st_time))

st_time = time.time()
for i in range(500):
  y = group_sort_3_sp(x)

ed_time = time.time()
print("naive g=3: {:.8f}".format(ed_time - st_time))

st_time = time.time()
for i in range(500):
  y = group_sort_3(x)
ed_time = time.time()
print("pytorch g=3: {:.8f}".format(ed_time - st_time))
