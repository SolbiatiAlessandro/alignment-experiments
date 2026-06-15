import torch
import numpy as np
from random import random

def data_loading_slow(x, batch_size, context_length, device='cpu'):
    xs, ys = [], []
    while len(xs) < batch_size: 
        prev = int(random() * (x.shape[0] - context_length))
        i = prev + context_length
        xs.append(torch.tensor(x[prev:i], device=device))
        ys.append(torch.tensor(x[prev+1:i+1], device=device))
    return torch.stack(xs), torch.stack(ys)


def data_loading_sampling(dataset, batch_size, context_length, device='cpu', dtype=torch.int):
    x_starts = np.random.randint(0, dataset.shape[0]-context_length, batch_size).reshape(-1, 1)
    offsets = np.arange(context_length).reshape(1, -1)
    x_index = offsets + x_starts
    y_index = x_index + 1
    return torch.tensor(dataset[x_index], device=device, dtype=dtype), torch.tensor(dataset[y_index], device=device, dtype=dtype)

def load_training_examples(train_examples, padding_token, max_context_length, device=None, dtype=torch.long):
    import numpy as np
    torch_train_examples = torch.ones((len(train_examples), max_context_length), device=device, dtype=dtype) * padding_token
    for i, e in enumerate(train_examples):
        torch_train_examples[i, 0:len(e)] = torch.tensor(e, device=device, dtype=dtype)
    return torch_train_examples

def data_loading(train_examples, batch_size, padding_token, device=None, dtype=torch.long):
    index = torch.randint(0, train_examples.shape[0], (batch_size,))
    x = train_examples[index]
    y = x.clone()
    loss_ignore_index = -100
    attention_padding = torch.ones_like(x, device=device, dtype=dtype)
    attention_padding[x == padding_token] = 0
    y[x == padding_token] = loss_ignore_index
    return x, y, attention_padding
