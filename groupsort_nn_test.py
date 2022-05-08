import torch
from groupsort.groupsort_cuda import GroupSort as CudaGroupSort
import time

class GroupSort(torch.nn.Module):
    def __init__(self, axis, group_size):
        super().__init__()
        self.axis = axis
        self.group_size = group_size

        assert axis == 1

    def forward(self, x):
        f = x.size(-1)
        return x.view(-1, self.group_size, f//self.group_size).sort(dim=1).values.view(-1, f)

class NN(torch.nn.Module):
    def __init__(self, in_features=100, out_features=100, hidden_features=100, n_layers=5, group_size=5, ours=False):
        super().__init__()

        layers = []
        prev_features = in_features
        for i in range(n_layers - 1):
            layers.append(torch.nn.Linear(prev_features, hidden_features))
            prev_features = hidden_features
        layers.append(torch.nn.Linear(prev_features, out_features))
        self.layers = torch.nn.ModuleList(layers)

        if ours:
            self.gs = CudaGroupSort(axis=1, group_size=group_size)
        else:
            self.gs = GroupSort(axis=1, group_size=group_size)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.gs(x)
        return self.layers[-1](x)

bs = 100000
in_features = 100
m_ours = NN(in_features=in_features, n_layers=5, group_size=5, ours=True)
m_ours.cuda()


m_torch = NN(in_features=in_features, n_layers=5, group_size=5, ours=False)
m_torch.cuda()
x = torch.randn(bs, in_features).cuda()


n_trials = 10


y = m_ours(x)
st_time = time.time()
for i in range(n_trials):
    y = m_ours(x)
ed_time = time.time() 
print("Ours Forward Pass time: {:.6f}".format((ed_time - st_time) / n_trials))


# y = m_torch(x)
# st_time = time.time()
# for i in range(n_trials):
#     y = m_torch(x)
# ed_time = time.time() 
# print("PyTorch Forward Pass time: {:.6f}".format((ed_time - st_time) / n_trials))

# st_time = time.time()
# for i in range(n_trials):
#     y = m_ours(x)
#     l = y.mean()
#     l.backward()
#     m_ours.zero_grad()
# ed_time = time.time() 
# print("Ours Forward Pass + Backward Pass time: {:.3f}".format((ed_time - st_time) / n_trials))

# st_time = time.time()
# for i in range(n_trials):
#     y = m_torch(x)
#     l = y.mean()
#     l.backward()
#     m_torch.zero_grad()
# ed_time = time.time() 
# print("PyTorch Forward Pass + Backward Pass time: {:.3f}".format((ed_time - st_time) / n_trials))
