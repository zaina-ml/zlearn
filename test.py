import zlearn as zl
from zlearn.nn import layers
from zlearn.nn import module

import time

start = time.time()

net = module.Sequential(
    layers.Linear(in_features=784, out_features=16),
    layers.ReLU(),
    layers.Linear(in_features=16, out_features=16),
    layers.ReLU(),
    layers.Linear(in_features=16, out_features=10),
)

dummy_image = zl.arange(start=0, step=1, stop=784, dtype=zl.Float32())

preds =  zl.softmax(net(dummy_image))

end = time.time()

print(f"zLearn time: {end - start}")

import torch
from torch import nn

start = time.time()


net = nn.Sequential(
    nn.Linear(in_features=784, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=10),
)

dummy_image = torch.arange(start=0, end=784, dtype=torch.float)

net.eval()

with torch.inference_mode():
    preds = torch.softmax(net(dummy_image), dim=0)

end = time.time()

print(f"PyTorch Time: {end - start}")

#zLearn time: 0.01182103157043457
#PyTorch Time: 0.001827239990234375

# Really cool! zLearn is about 10x slower than PyTorch, showing how optimized PyTorch is!