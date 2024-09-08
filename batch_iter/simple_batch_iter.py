import torch


class BatchIter:
    """
    A simple iterator that returns mini-batches from a set of tensors.
    """

    def __init__(self, *tensors: torch.Tensor,
                 batch_size: int,
                 shuffle: bool = True):
        """
        tensors: feature tensors (each with shape: num_instances x *)
        batch_size: mini-batch size
        shuffle: whether to shuffle the tensors
        """
        self.tensors = tensors

        device = tensors[0].device
        n = tensors[0].size(0)
        if shuffle:
            indices = torch.randperm(n, device=device)
        else:
            indices = torch.arange(n, device=device)

        self.indices = indices.split(batch_size)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        tensors = self.tensors
        for batch_indices in self.indices:
            yield tuple((x[batch_indices, ...] for x in tensors))
