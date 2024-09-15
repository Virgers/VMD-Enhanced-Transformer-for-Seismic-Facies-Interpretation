import os
import torch
import datetime

from ITransformer_cls_options import Options, setup

device = torch.device('cuda:0')
torch.cuda.set_device(device)
os.environ['CUDA_VISIBLE_DEVICES'] ="0"


def main(config):
    # We have other tasks, for simplicty only keep classifcation here
    if args.task_name =='classification':
        from exp.exp_classification import Exp_Classification
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
            print('>>Start training : {}>>'.format(formatted_setting))
            exp.train(setting)
            torch.cuda.empty_cache()

    if args.is_testing:
        args.model_id = 'Test' 
        setting = {
                "datetime": formatted_date_time,
                "model": args.model,
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