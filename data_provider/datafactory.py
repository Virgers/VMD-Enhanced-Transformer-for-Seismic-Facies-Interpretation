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
    
