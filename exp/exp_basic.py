# Model Experiment Base
# Quick Start
# Inherit the base class:

# Create your experiment class inheriting from Exp_Basic
# Implement required methods: _build_model(), _get_data(), train(), vali(), test()
# Initialize with configuration:

# from exp.exp_basic import Exp_Basic

# class MyExperiment(Exp_Basic):
#     def _build_model(self):
#         # Select a model from self.model_dict based on args
#         return self.model_dict[self.args.model](self.args)
    
#     # Implement other required methods...

# # Run your experiment
# exp = MyExperiment(args)
# exp.train()
# Use provided functionality:

# Model registry with multiple transformer architectures
# Automatic device selection (CPU/GPU)
# Base experiment structure
# Output
# Base class methods: Device configuration, model initialization
# Model dictionary: Access to multiple transformer architectures
# Abstract methods: Structure for implementing custom experiment logic

import os
import torch

from models import (Autoformer, BiLSTM, FEDformer, Informer,
                    Nonstationary_Transformer, Pyraformer, Reformer, TimesNet,
                    Transformer, iTransformer)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Pyraformer': Pyraformer,
            'iTransformer': iTransformer,
            'BiLSTM': BiLSTM
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    # GPU useages
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
