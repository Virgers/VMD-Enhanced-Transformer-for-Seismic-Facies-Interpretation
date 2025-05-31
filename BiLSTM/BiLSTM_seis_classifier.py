import os
import torch
import datetime

from BiLSTM.BiLSTM_seis_cls_cfg import Options, setup

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device('cuda:0')
torch.cuda.set_device(device)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main(config):
    from BiLSTMexpcls import Exp_Classification
    Exp = Exp_Classification
   
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.is_training:
        for ii in range(args.itr):
            args.model_id = 'Train' 
            setting = {
                'timestamp': formatted_date_time,
                'model': args.model,
                'dataset': args.dataset,
                'train_prop': args.train_proportion,
                'test_prop': args.test_proportion,
                'mask_rate': args.mask_rate,
                'is_vmd': args.is_vmd,
                'embed_flag': args.embedding_flag,
                'epochs': args.train_epochs,
                'batch_size': args.batch_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'iteration': ii
            }
            
            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            torch.cuda.empty_cache()

    if args.is_testing:
        args.model_id = 'test' 
        setting = {
            'timestamp': formatted_date_time,
            'model': args.model,
            'dataset': args.dataset,
            'test_prop': args.test_proportion,
            'mask_rate': args.mask_rate,
            'is_vmd': args.is_vmd,
            'embed_flag': args.embedding_flag,
            'epochs': args.train_epochs,
            'batch_size': args.batch_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers
        }

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=0)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    args = Options().parse()
    os.makedirs(args.output_dir, exist_ok=True)
    config = setup(args)
    main(config)