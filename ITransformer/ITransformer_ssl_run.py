import os
import torch
import datetime

from ITransformer_ssl_options import Options, setup
from exp.exp_ssl import Exp_ssl

device = torch.device('cuda:0')
torch.cuda.set_device(device)
os.environ['CUDA_VISIBLE_DEVICES'] ="0"


def main(config):
       
    Exp = Exp_ssl

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y%m%d_%H%M")
    
    if args.is_training:
        args.mode = 'Train'
        for ii in range(args.itr):
            setting = {
                "mode":f"{args.task_name}_{args.mode}",
                "datetime": formatted_date_time,
                "dataset": args.dataset,
                "train_prop": f"{args.train_proportion:.2f}",
                "test_prop": f"{args.test_proportion:.2f}",
                "mask_rate": f"{args.mask_rate:.2f}",
                "is_vmd": str(args.is_vmd),
                "embed": args.embedding_flag,
                "epochs": str(args.train_epochs),
                "batch": str(args.batch_size)
                }
            
            exp = Exp(args)
            formatted_setting = ' | '.join([f"{k}:{v}" for k, v in setting.items()])
            print('>>Start ssl : {}>>'.format(formatted_setting))
            exp.train(setting)
            torch.cuda.empty_cache()

    if args.is_testing:
        args.mode = 'Test' 
        setting = {
                "mode":f"{args.task_name}_{args.mode}",
                "datetime": formatted_date_time,
                "dataset": args.dataset,
                "test_prop": f"{args.test_proportion:.2f}",
                "mask_rate": f"{args.mask_rate:.2f}",
                "is_vmd": str(args.is_vmd),
                "embed": args.embedding_flag,
                }

        exp = Exp(args)  # set experiments
        formatted_setting = ' | '.join([f"{k}:{v}" for k, v in setting.items()])
        print('>>testing : {}<<'.format(formatted_setting))
        exp.test(setting, test=0)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    
    args = Options().parse() 
    config = setup(args) 
    main(config)