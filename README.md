# Pytorch batch iteration utilities

Utilities for iterating over mini-batches of tensors, while avoiding the overhead incurred by the significant DataLoader class when training light-weight models. 

Installing:

```bash
pip install batch_iter
```

The library is based on this [blog post](https://alexshtf.github.io/2024/08/31/BatchIter.html). There you can find more explanation and motivation. Below are simple examples:

## Plain iteration

```python
from batch_iter import BatchIter
import torch

X = torch.arange(15).reshape(5, 3)
y = torch.linspace(-1, 1, 5)

for Xb, yb in BatchIter(X, y, batch_size=2, shuffle=True):
  print('---')
  print('features = ', Xb)
  print('labels = ', yb)
```

The output is:

```
---
features =  tensor([[12, 13, 14],
        [ 6,  7,  8]])
labels =  tensor([1., 0.])
---
features =  tensor([[3, 4, 5],
        [0, 1, 2]])
labels =  tensor([-0.5000, -1.0000])
---
features =  tensor([[ 9, 10, 11]])
labels =  tensor([0.5000])
```

## Grouped iteration

The library also supports iterating over groups of tensors, identified by an additional group-id tensor. This is useful, for example, when training a ranking model, and we would like to iterate over mini-batches consisting of full queries. Each query is a group. 

For example:

```python
from batch_iter import GroupBatchIter
import torch

X = torch.arange(15).reshape(5, 3)
y = torch.linspace(-1, 1, 5)
# first three samples are a group with id 1, 
# the next two samples are another group with id 2.
group_id = torch.tensor([1, 1, 1, 2, 2])

for gb, Xb, yb in GroupBatchIter(group_id, X, y, batch_size=2, shuffle=True):
  print('---')
  print('group_id = ', gb)
  print('features = ', Xb)
  print('labels = ', yb)
```

The output is:

```
---
group_id =  tensor([1, 1, 1, 2, 2])
features =  tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14]])
labels =  tensor([-1.0000, -0.5000,  0.0000,  0.5000,  1.0000])
```

The entire data-set is one mini-batch, since we chose a mini-batch of size two, meaning *two groups*.
