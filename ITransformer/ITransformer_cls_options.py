import torch    
import argparse
import random

import logging
import sys
import os
import traceback
import json
from datetime import datetime
import random
import string
import logging
import numpy as np
logger = logging.getLogger('__main__')


class Options(object):

    def __init__(self):
        fix_seed = 2023
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        self.data ='custom'
        self.parser = argparse.ArgumentParser(
            description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')
        
        self.parser = argparse.ArgumentParser(description='TimesNet')
        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')
        self.parser.add_argument('--output_dir', default='/home/dell/disk1/Jinlong/Time-Series-Library-main/output', 
                                 help='Root output directory. Must exist. Time-stamped directories will be created inside.')
        self.parser.add_argument('--dataset', type=str, default='f3')
        self.parser.add_argument('--is_training', type=int,  default=1, help='status')
        self.parser.add_argument('--is_pre_training', type=int,  default=0, help='status')
        self.parser.add_argument('--is_self_supervised', type=int,  default=0, help='status')
        self.parser.add_argument('--is_testing', type=int,  default=0, help='status')
        self.parser.add_argument('--task_name', type=str, default='classification')
        self.parser.add_argument('--model', type=str,  default='iTransformer',
                            help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, BiLSTM]')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        self.parser.add_argument('--val_proportion', type=float, default=0.001, help='seismic data for test')
        self.parser.add_argument('--train_proportion', type=float, default=0.00001, help='seismic data for train')
        self.parser.add_argument('--test_proportion', type=float, default=0.0001, help='seismic data for test')

        # F3dataset configuration 
        self.parser.add_argument('--root_path', type=str, default='/home/dell/disk1/Jinlong/faciesdata', help='root path of the data file')
        self.parser.add_argument('--data_path', type=str, default='train_seismic.npy', help='data npy file')
        self.parser.add_argument('--label_path', type=str, default='train_labels.npy', help='label npy file')
        self.parser.add_argument('--mask_path', type=str, default='train_labels.npy', help='mask npy file')
        
        # Parihaka dataset configuration
        # self.parser.add_argument('--root_path', type=str, default='/home/dell/disk1/Jinlong/faciesdata', help='root path of the data file')
        # self.parser.add_argument('--data_path', type=str, default='data_train.npz', help='data npy file')
        # self.parser.add_argument('--label_path', type=str, default='labels_train.npz', help='label npy file')
        # self.parser.add_argument('--mask_path', type=str, default='labels_train.np
        # z', help='mask npy file')

        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--vmd_data_path', type=str, default='full_F3_vmd.npy', help='vmd npy data file') 
        self.parser.add_argument('--is_vmd', type=bool, default=False , help='whether using the vmd')
        self.parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
        self.parser.add_argument('--embedding_flag', type=bool, default=False, help='if set true model apply padding mask as attn mask')
        self.parser.add_argument('--mask_rate', type=int, default=0.125, help='mask rate below is masked')

        self.parser.add_argument('--checkpoints_test_only', type=str, default='/home/dell/disk1/Jinlong/Time-Series-Library-main/checkpoints/2024-06-13 12:26:26-iTransformer-trainproportion_0.01-testproportion_0.01-maskrate_0.125-isvmd_False-embedflag_False-epochs_14-backbone_False-batchsize_16-0/checkpoint.pth', help='location of model checkpoints')

        self.parser.add_argument('--model_id', type=str,  default='train', help='model id')
        # data loader
        self.parser.add_argument('--data', type=str, default='TSF', help='dataset type')

        self.parser.add_argument('--attn_mask_path', type=str, default='labels.npy', help='mask npy file')

        self.parser.add_argument('--is_attn_mask', type=bool, default=False, help='if set true model apply padding mask as attn mask')
        self.parser.add_argument('--att_mask_none', type=bool, default=True, help='if set true model apply padding mask as attn mask')

        self.parser.add_argument('--val_test_proportion', type=float, default=0.005, help='seismic data for test')
        self.parser.add_argument('--self_supervised_proportion', type=float, default=0.005, help='seismic data for test')
        self.parser.add_argument('--max_features', type=int, default=255, help='Vocabulary size')
        self.parser.add_argument('--embed_size', type=int, default=64, help='Embedding dimension')
        self.parser.add_argument('--num_class', type=int, default=6, help='Embedding dimension')

        self.parser.add_argument('--num_class2', type=int, default=6, help='Embedding dimension')
        
        self.parser.add_argument('--features', type=str, default='S',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        self.parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        # OT is the dataset last column represents the label
        self.parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

        # forecasting task
        self.parser.add_argument('--seq_len', type=int, default=255, help='input sequence length')
        self.parser.add_argument('--seq_len2', type=int, default=1006, help='input sequence length')
        self.parser.add_argument('--label_len', type=int, default=100, help='start token length') # no longer needed in inverted Transformers
        self.parser.add_argument('--pred_len', type=int, default=255, help='prediction sequence length')

        # model define
        self.parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
        self.parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        self.parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
        self.parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
        self.parser.add_argument('--c_out', type=int, default=6, help='output size')
        self.parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
        self.parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        self.parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        self.parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        self.parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
        self.parser.add_argument('--moving_avg', type=int, default=5, help='window size of moving average')
        self.parser.add_argument('--factor', type=int, default=1, help='attn factor')
        self.parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        self.parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation')
        self.parser.add_argument('--output_attention', type=bool,default=True, help='whether to output attention in ecoder')
        # optimization
        self.parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
        self.parser.add_argument('--itr', type=int, default=1, help='experiments times')
        self.parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        self.parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
        self.parser.add_argument('--des', type=str, default='test', help='exp description')
        self.parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        self.parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        self.parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU 
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        self.parser.add_argument('--exp_name', type=str, default='MTSF',
                            help='experiemnt name, options:[MTSF, partial_train]')
        self.parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
        self.parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        self.parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
        self.parser.add_argument('--target_root_path', type=str, default=r'F:\Facies\all_datasets\weather', help='root path of the data file')
        self.parser.add_argument('--target_data_path', type=str, default='weather.csv', help='data file')
     
        self.parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
        self.parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
        self.parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                            'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    
    def parse(self):
        
        args = self.parser.parse_args()
    
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        print('Args in experiment:')
        print(args)

        
        return args


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['exp_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (len(config['exp_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config

