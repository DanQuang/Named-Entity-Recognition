import torch
from typing import List, Optional

def padding(array, max_length, padding_value):
    """
    Input:
        array: list of words in the sequence that need padding
        max_length: output of word list after padding
        padding_value: padding value
    Output:
        a tensor of the sequence after padding
    """
    if len(array) < max_length:
            padding_length = max_length - len(array)
            padding_array = array + [padding_value]*padding_length
            return torch.tensor(padding_array, dtype= torch.long)
    else:
        return torch.tensor(array[:max_length], dtype= torch.long)
