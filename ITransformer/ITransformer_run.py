import os
import torch
import datetime

from ITransformer_cls_options import Options, setup

device = torch.device('cuda:0')
torch.cuda.set_device(device)
os.environ['CUDA_VISIBLE_DEVICES'] ="0"

# device = torch.device('cuda:1')
# torch.cuda.set_device(device)
# os.environ['CUDA_VISIBLE_DEVICES'] ="1"

def main(config):
    # we have other tasks, for simplicty only keep classifcation here
    if args.task_name =='classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    if args.is_training:
        for ii in range(args.itr):
            args.model_id = 'Train' 
            setting = (
                f"{formatted_date_time}-"
                f"{args.model}-"
                f"dataset_{args.dataset}-"
                f"train-proportion_{args.train_proportion}-"
                f"test-proportion_{args.test_proportion}-"
                f"mask-rate_{args.mask_rate}-"
                f"is-vmd_{args.is_vmd}-"
                f"embed-flag_{args.embedding_flag}-"
                f"epochs_{args.train_epochs}-"
                f"backbone_{args.use_ssl_model}-"
                f"batchsize_{args.batch_size}-"
                f"{ii}"
            )
            
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)
            
            torch.cuda.empty_cache()


    if args.is_testing:
        args.model_id = 'Test' 
        setting = (
            f"{formatted_date_time}-"
            f"{args.model}-"
            f"dataset_{args.dataset}-"
            f"test-proportion_{args.test_proportion}-"
            f"mask-rate_{args.mask_rate}-"
            f"is-vmd_{args.is_vmd}-"
            f"embed-flag_{args.embedding_flag}-"
            f"epochs_{args.train_epochs}-"
            f"batchsize_{args.batch_size}-"
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=0)
        torch.cuda.empty_cache()


if __name__ == '__main__':

    args = Options().parse() 
    config = setup(args) 
    main(config)