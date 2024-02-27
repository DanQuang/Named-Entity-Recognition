import torch
from typing import List, Optional

def padding_sequence(array, max_length, padding_value):
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
            return torch.tensor(padding_array, dtype= torch.int)
    else:
        return torch.tensor(array[:max_length], dtype= torch.int)
    
def padding_tags(list_tags, max_length, padding_value):
    out = []
    for tags in list_tags:
        if len(tags) < max_length:
            padding_length = max_length - len(tags)
            padding_array = tags + [padding_value]*padding_length
            out.append(padding_array)
        else:
             out.append(tags[:max_length])
    return torch.tensor(out, dtype= torch.int16)
