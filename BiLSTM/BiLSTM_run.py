import os
import torch
import datetime
# from test_exp_classification import Exp_Classification

# from ITransformer_imp_options import Options, setup
from BiLSTMoptions import Options, setup
# from ITransformer_nz_cls_options import Options, setup

num_gpus = torch.cuda.device_count()
# print(f"Number of available GPUs: {num_gpus}")
# List GPU names
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device('cuda:0')
torch.cuda.set_device(device)

os.environ['CUDA_VISIBLE_DEVICES'] ="0"

# device = torch.device('cuda:1')
# torch.cuda.set_device(device)

# os.environ['CUDA_VISIBLE_DEVICES'] ="1"

def main(config):
    

    from BiLSTMexpcls import Exp_Classification
    Exp = Exp_Classification
   

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # Exp = Exp_Classification
    if args.is_self_supervised:
            for ii in range(args.itr):
                args.model_id = 'self_supervised'
                setting = (
                    f"{formatted_date_time}-"
                    f"{args.model}-"
                    f"trainproportion_{args.train_proportion}-"
                    f"testproportion_{args.test_proportion}-"
                    f"masktype_{args.mask_type}-"
                    f"maskrate_{args.mask_rate}-"
                    f"isvmd_{args.is_vmd}-"
                    f"embedflag_{args.embedding_flag}-"
                    f"epochs_{args.train_epochs}-"
                    f"backbone_{args.use_ssl_model}-"
                    f"batchsize_{args.batch_size}-"
                    f"{ii}"
                )
                exp = Exp(args)  # set experiments
                print('>>>>>>>start self learning training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
                
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                # exp.self_supervised(setting)
    if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                
                args.model_id = 'Train' 
                setting = (
                    f"{formatted_date_time}-"
                    f"{args.model}-"
                    f"dataset_{args.dataset}-"
                    f"trainproportion_{args.train_proportion}-"
                    f"testproportion_{args.test_proportion}-"
                    f"maskrate_{args.mask_rate}-"
                    f"isvmd_{args.is_vmd}-"
                    f"embedflag_{args.embedding_flag}-"
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


    if args.is_pre_training:
            for ii in range(args.itr):
                # setting record of experiments
                args.model_id ='Train'
                
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_des{}_isvmd{}_attnmask{}_att_mask_none{}_embedding_flag{}'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    args.is_vmd,
                    args.is_attn_mask,
                    args.att_mask_none,
                    args.embedding_flag,
                    args.class_strategy, ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>start pre training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.pre_train(setting)

                # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting)
                # # if args.do_predict:
                #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                #     exp.predict(setting, True)
                
                torch.cuda.empty_cache()

    if args.is_testing:
        args.model_id = 'test' 
        setting = (
            f"{formatted_date_time}-"
            f"{args.model}-"
            f"dataset_{args.dataset}-"
            f"testproportion_{args.test_proportion}-"
            f"maskrate_{args.mask_rate}-"
            f"isvmd_{args.is_vmd}-"
            f"embedflag_{args.embedding_flag}-"
            f"epochs_{args.train_epochs}-"
            f"batchsize_{args.batch_size}-"
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=0)
        torch.cuda.empty_cache()


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)