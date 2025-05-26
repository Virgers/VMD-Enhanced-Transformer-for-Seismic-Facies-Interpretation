# Seismic Data Provider
# Quick Start
# Prepare your data:

# Organize your data files (seismic, VMD, labels) in appropriate directories
# Configure root_path, data_path, vmd_data_path, and label_path in your args
# Set train/test/validation proportions as needed
# Configure mask_path if using masked data
# Import and use:

# from data_provider.datafactory import data_provider

# # For standard data
# dataset, dataloader = data_provider(args, is_vmd=False, flag='train')

# # For VMD-processed data
# dataset, dataloader = data_provider(args, is_vmd=True, flag='test')
# Access data batches:

# for batch in dataloader:
#     # Process your batch data
#     inputs, labels = batch
# Output
# dataset: Dataset object containing all data samples
# dataloader: PyTorch DataLoader configured for training/testing
# Each batch contains seismic data (with optional VMD decomposition) and labels
# Configurable batch size, worker count, and shuffle settings

from torch.utils.data import DataLoader
from data_provider.dataloader import TSF_custom
from data_provider.uea import collate_fn

data_dict = {
    'TSF':TSF_custom
}

def data_provider(args, is_vmd, flag):
    Data = data_dict[args.data]
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size  # bsz for train and valid

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
       
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid  

    drop_last = False
    data_set = Data(
        is_vmd,
        flag,
        root_path=args.root_path,
        data_path= args.data_path,
        vmd_data_path = args.vmd_data_path,
        label_path= args.label_path,
        mask_path = args.mask_path,
        train_proportion = args.train_proportion,
        test_proportion = args.test_proportion,
        val_proportion = args.val_proportion,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader
    
