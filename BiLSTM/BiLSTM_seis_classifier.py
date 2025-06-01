import datetime
import os
import torch
from BiLSTM.BiLSTM_seis_cls_cfg import Options, setup

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device('cuda:0')
torch.cuda.set_device(device)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main(config):
    from exp.BiLSTMexpcls import Exp_Classification
    Exp = Exp_Classification

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y%m%d_%H%M")
    
    if args.is_training:
        for ii in range(args.itr):
            args.model_id = 'Train' 
            setting = {
                "datetime": formatted_date_time,
                "model": args.model,
                "dataset": args.dataset,
                "train_prop": f"{args.train_proportion:.3f}",
                "test_prop": f"{args.test_proportion:.3f}",
                "mask_rate": f"{args.mask_rate:.3f}",
                "is_vmd": str(args.is_vmd),
                "embed": str(args.embedding_flag),
                "epochs": str(args.train_epochs),
                "batch": str(args.batch_size),
                "hidden": str(args.hidden_size),
                "layers": str(args.num_layers),
                "iter": str(ii)
            }
            
            exp = Exp(args)
            formatted_setting = ' | '.join([f"{k}:{v}" for k, v in setting.items()])
            print('>>Start training : {}>>'.format(formatted_setting))
            exp.train(setting)
            
            formatted_setting = ' | '.join([f"{k}:{v}" for k, v in setting.items()])
            print('>>testing : {}<<'.format(formatted_setting))
            exp.test(setting)
            
            torch.cuda.empty_cache()

    if args.is_testing:
        args.model_id = 'Test' 
        setting = {
            "datetime": formatted_date_time,
            "model": args.model,
            "dataset": args.dataset,
            "test_prop": f"{args.test_proportion:.3f}",
            "mask_rate": f"{args.mask_rate:.3f}",
            "is_vmd": str(args.is_vmd),
            "embed": str(args.embedding_flag),
            "hidden": str(args.hidden_size),
            "layers": str(args.num_layers)
        }

        exp = Exp(args)
        formatted_setting = ' | '.join([f"{k}:{v}" for k, v in setting.items()])
        print('>>testing : {}<<'.format(formatted_setting))
        exp.test(setting, test=0)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    args = Options().parse()
    os.makedirs(args.output_dir, exist_ok=True)
    config = setup(args)
    main(config)