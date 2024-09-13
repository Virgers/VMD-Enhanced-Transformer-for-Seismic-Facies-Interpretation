import os
import pandas as pd
import numpy as np
import torch

def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    # original inputs has shape of  (seq_length, feat_dim);
    # seismic inputs train_data has shape of 
    data_all, label_all, padding_masks, attn_masks = zip(*data)
   
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in data_all]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    # we asure the max_len is same as the data incase the last
    # max_len = max_len - 1
    X = torch.zeros(batch_size, max_len, data_all[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        
        X[i, :end, :] = data_all[i][:end, :]

    Y = torch.zeros(batch_size, max_len, label_all[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        
        Y[i, :end, :] = label_all[i][:end, :]
  
    mask = torch.zeros(batch_size, max_len, padding_masks[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        
        mask[i, :end, :] = padding_masks[i][:end, :]

    attn_mask = torch.zeros(batch_size, max_len, attn_masks[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        
        attn_mask[i, :end, :] = attn_masks[i][:end, :]

    return X, Y, mask, attn_mask
    # return X, Y