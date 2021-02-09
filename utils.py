import torch

def pad_tensor(vec, length, dim, pad_symbol):
    pad_size = list(vec.shape)
    pad_size[dim] = length - vec.shape[dim]
    answer = torch.cat([vec, torch.ones(*pad_size, dtype=torch.long, device=vec.device) * pad_symbol], axis=dim)
    return answer

def pad_tensors(tensors, pad=0):
    try:
        if tensors[0].ndim > 0:
            L = max(tensor.shape[0] for tensor in tensors)
            tensors = [pad_tensor(tensor, L, dim=0, pad_symbol=pad) for tensor in tensors]
        return torch.stack(tensors, axis=0)
    except:
        return tensors

def collate_fn(batch):
    if isinstance(batch[0], list):
        return [collate_fn(elem[i]) for i in range(len(batch[0]))]
    elif isinstance(batch[0], dict):
        return {key: collate_fn([elem[key] for elem in batch]) for key in batch[0]}
    return pad_tensors(batch)