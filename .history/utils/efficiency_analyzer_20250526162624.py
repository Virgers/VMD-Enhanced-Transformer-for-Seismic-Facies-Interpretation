# Model Efficiency Analyzer
# Quick Start
# Import the tool:

# from utils.model_efficiency_analyzer import main
# Run analysis on various models:

# # For a single model and mask rate
# main('ITransformer', 0.25)

# # For multiple models and mask rates
# models = ['NST', 'FED', 'ITransformer']
# mask_rates = [0.125, 0.25, 0.375, 0.5]

# for model in models:
#     for mask_rate in mask_rates:
#         main(model, mask_rate)
# Interpret results:

# The tool will output the best model configuration based on performance vs. training time
# Results show MSE/MAE improvements normalized by computational cost
# Output
# Best setup index: The optimal model configuration
# Training time: Seconds required for the selected configuration
# Cost-benefit ratio: Performance gain per unit of computational time
# Frequency analysis: Most commonly selected configurations across runs

import os
import sys
from collections import OrderedDict
from textwrap import wrap

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.iTransformer import Model
from collections import OrderedDict
from textwrap import wrap

import numpy as np
import pandas as pd
import torch


def analyze_model_structure(model, save_path='model_analysis'):
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    """
    Analyzes PyTorch model structure and saves results in multiple formats
    
    Args:
        model: PyTorch model
        save_path: Base path for saving output files (without extension)
    
    Returns:
        dict: Summary of model statistics
    """
    # Initialize containers
    layer_info = []
    total_params = 0
    total_trainable = 0
    
    # Analyze each named module
    for name, module in model.named_modules():
        # Skip container modules
        if list(module.children()):
            continue
            
        # Get parameter count for this layer
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Get input/output shape if available
        input_shape = "N/A"
        output_shape = "N/A"
        if hasattr(module, 'in_features'):
            input_shape = module.in_features
        elif hasattr(module, 'in_channels'):
            if hasattr(module, 'kernel_size'):
                input_shape = f"{module.in_channels}×{module.kernel_size}"
            else:
                input_shape = module.in_channels
                
        if hasattr(module, 'out_features'):
            output_shape = module.out_features
        elif hasattr(module, 'out_channels'):
            if hasattr(module, 'kernel_size'):
                output_shape = f"{module.out_channels}×{module.kernel_size}"
            else:
                output_shape = module.out_channels
        
        # Store layer information
        layer_info.append({
            'Layer': name if name else 'model',
            'Type': module.__class__.__name__,
            'Input Shape': input_shape,
            'Output Shape': output_shape,
            'Parameters': params,
            'Trainable': trainable
        })
        
        total_params += params
        total_trainable += trainable
    
    # Create DataFrame
    df = pd.DataFrame(layer_info)
    
    # Save as CSV
    df.to_csv(f'{save_path}.csv', index=False)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, 
                             float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x,
                             caption='Model Architecture and Parameters',
                             label='tab:model_structure')
    
    with open(f'{save_path}.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate Markdown table
    markdown_table = df.to_markdown(index=False)
    
    with open(f'{save_path}.md', 'w') as f:
        f.write(markdown_table)
    
    # Generate summary statistics
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': total_trainable,
        'non_trainable_parameters': total_params - total_trainable,
        'layer_count': len(layer_info)
    }
    
    # Save summary
    with open(f'{save_path}_summary.txt', 'w') as f:
        f.write('Model Summary:\n')
        f.write(f'Total Parameters: {total_params:,}\n')
        f.write(f'Trainable Parameters: {total_trainable:,}\n')
        f.write(f'Non-trainable Parameters: {total_params - total_trainable:,}\n')
        f.write(f'Number of Layers: {len(layer_info)}\n')
    
    return summary

# print diformer


class Config:
    task_name = 'classification'
    seq_len = 255  # Based on your reshape operation
    pred_len = 0   # Not used in classification
    output_attention = True
    is_vmd = False  # Based on default values
    embedding_flag = True
    d_model = 64   # From your embedding dimension
    enc_in = 1     # Input channels
    embed = 'fixed'
    freq = 'h'
    dropout = 0.1
    factor = 5
    n_heads = 8
    e_layers = 2
    d_ff = 256
    activation = 'gelu'
    num_class2 = 1  # Based on your output reshape

# Create model instance
configs = Config()
model = Model(configs)

# total_feature_dim = 14
# num_classes = 7
# fusion_height = 16
# fusion_width = 288

# # print UNET
# model = MemoryEfficientUNetFusion(
#     total_feature_dim=total_feature_dim,
#     num_classes=num_classes,
#     fusion_height=fusion_height,
#     fusion_width=fusion_width
# )

summary = analyze_model_structure(model, save_path='./output/thesis_model_analysis/iformer_model_analysis')
