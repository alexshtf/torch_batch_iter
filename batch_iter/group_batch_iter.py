import torch


def lexsort(*keys: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """ Computes the lexicographical sorting order of the given tensors, starting from the least significant to the
        most significant ones
    Args:
        keys: list of torch.Tensor, the tensors to sort
        dim: int, the dimension along which to sort
    Returns:
        torch.Tensor, the sorting order
    """
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


def view_as_bytes(x: torch.Tensor) -> [torch.Tensor]:
    """
    Decomposes the given tensor to its constituent bytes. The byte order depends on the computer architecture,
    and is not consistent. So the result isn't useful for persistence, just for in-memory computations. The bytes
    are added as the last dimension.
    Args:
        x: torch.Tensor, the tensor to decompose
    Returns:
        list of torch.Tensor of type uint8, the decomposed bytes
    """
    element_bytes = x.dtype.itemsize
    bytes_tensor = x.view(torch.uint8).view(x.shape + (element_bytes,))
    return bytes_tensor.unbind(dim=-1)


def fnv_hash(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the FNV hash for each component of a PyTorch tensor of integers.
    Args:
      tensor: torch.Tensor the tensor for which we compute element-wise FNV hash
    Returns:
      A PyTorch tensor of the same size and dtype as the input tensor, containing the FNV hash for each element.
    """
    # Define the FNV prime and offset basis
    fnv_prime = torch.tensor(0x01000193, dtype=torch.int32, device=tensor.device)
    fnv_offset = -2128831035  # 0x811c9dc5 as a signed int32

    # Initialize the hash value with zeros (same size and dtype as tensor)
    hash_value = torch.full_like(tensor, fnv_offset, dtype=torch.int32)
    for byte in view_as_bytes(tensor):
        hash_value = torch.bitwise_xor(hash_value, byte)
        hash_value = hash_value * fnv_prime

    # No need to reshape, output already has the same size and dtype as input
    return hash_value


def group_idx(group_id: torch.Tensor) -> torch.Tensor:
    """
    Given a sequence of group ids, each group given in consecutive order, compute the index where each
    group begins.
    Args:
      group_id: torch.Tensor, shape (N,) the group ids
    Returns:
      torch.Tensor, shape (N+1,) the index where each group begins. The last element is the length of the
    """
    values, counts = group_id.unique_consecutive(return_counts=True)
    idx = torch.cumsum(counts, dim=-1)
    return torch.nn.functional.pad(idx, (1, 0))


def batch_endpoint_indices(group_idx: torch.Tensor, batch_size: int) -> ([int], [int]):
    """
    Given a tensor of indices where each group begins, and a batch size - compute
    the start and end points of each mini-batch, each consisting of the specified
    number of groups
      Args:
          group_idx: torch.Tensor, shape (N+1,) the index where each group begins
          batch_size: int, the number of groups in each batch
      Returns:
          tuple of two lists, the start and end points of each batch
    """
    # pad group_idx to the smallest multiple of batch_size
    padding_size = batch_size - (len(group_idx) - batch_size * (len(group_idx) // batch_size))
    if padding_size > 0:
        padding = group_idx[-1].expand(padding_size)
        group_idx = torch.cat((group_idx, padding), dim=-1)

    # extract start and end points
    start_points = group_idx[0:-1:batch_size]
    end_points = group_idx[batch_size::batch_size]

    # return them as a list, so we can iterate over them
    return start_points.tolist(), end_points.tolist()


class GroupBatchIter:
    """
    Given a group id tensor, and a set of additional tensors, this iterator
    yields mini-batches of the specified size, where each mini-batch contains
    the group id, and the additional tensors. The order within the groups is
    preserved, but the groups may be shuffled.

    Shuffling order does not use the PyTorch random seed, and it is recommended
    to use a different seed for each epoch, such as the epoch number.
    """

    def __init__(self, group_id: torch.Tensor,
                 *tensors: torch.Tensor,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 shuffle_seed: int = 42):
        """
        Args:
          group_id: torch.Tensor, the group id tensor
          tensors: list of torch.Tensor, the additional tensors
          batch_size: int, the number of groups in each batch
          shuffle: bool, whether to shuffle the groups
          shuffle_seed: int, the seed for the shuffle
        """
        self.group_id = group_id
        self.tensors = tensors

        if shuffle:
            self.idxs = lexsort(group_id, fnv_hash(group_id + shuffle_seed))
        else:
            self.idxs = torch.arange(len(group_id), device=group_id.device)

        group_start_indices = group_idx(group_id[self.idxs])
        self.batch_start, self.batch_end = batch_endpoint_indices(group_start_indices, batch_size)

    def __len__(self):
        return len(self.batch_start)

    def __iter__(self):
        # we create mini-batches containing both group-id, and the additional
        # tensors
        tensors = (self.group_id,) + self.tensors

        # iterate over batch endpoints, and yield tensors
        for start, end in zip(self.batch_start, self.batch_end):
            batch_idxs = self.idxs[start:end]
            if len(batch_idxs) > 0:
                yield tuple(x[batch_idxs, ...] for x in tensors)
