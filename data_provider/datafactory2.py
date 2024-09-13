# from test_dataloader import TSF_custom
from torch.utils.data import DataLoader

from data_provider.dataloader2 import TSF_custom
from data_provider.uea2 import collate_fn

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
        if args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid  

    if args.task_name == 'classification' or args.task_name == 'self_supervised' :
        # print()
        drop_last = False
        data_set = Data(
            is_vmd,
            flag,
            root_path=args.root_path,
            data_path = args.data_path,
            vmd_data_path = args.vmd_data_path,
            label_path=args.label_path,
            mask_path = args.mask_path,
            attn_mask_path = args.attn_mask_path,
            train_proportion = args.train_proportion,
            test_proportion = args.test_proportion,
            val_proportion = args.val_proportion,
            val_test_proportion = args.val_test_proportion,
            self_supervised_proportion = args.self_supervised_proportion
        )

        # here the collate_fn gives the padding_mask
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=2,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len2)
        )

        return data_set, data_loader
    
