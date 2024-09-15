import os
import pdb
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')

   
class Exp_Classification(Exp_Basic):
   
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
       

    def _build_model(self):

        # model init load pretrained model 
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
         
        cls_projection = nn.Linear(self.args.d_model * self.args.enc_in,  self.args.num_class2)
        model.projection = cls_projection

        return model
        
    def _get_data(self, flag, is_vmd, dataset):
        # It may take a period of time to load VMD data
         
        if dataset=='f3':
            from data_provider.datafactory import data_provider
        else:
            from data_provider.datafactory2 import data_provider

        data_set, data_loader = data_provider(self.args, is_vmd, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
                    
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()

        return criterion

    def vali(self,  criterion):
        total_loss = []
        accuracy = 0
        total_samples = 0
        total_correct = 0

        # print('---------------Get validation dataset----------')

        _, vali_loader = self._get_data(is_vmd=self.args.is_vmd, flag='val',dataset=self.args.dataset)

        print('---------------Get validation dataset done---------')

        self.model.eval()

        with torch.no_grad():
            
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs, _ = self.model(batch_x, padding_mask)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)
                probs = torch.nn.functional.softmax(pred)  # (total_samples, num_classes) est. prob. for each class and sample
                predictions = torch.argmax(probs, dim=1) # (total_samples,) int class index for each sample
                total_correct += (predictions.cpu() == label.cpu()).sum().item()
                total_samples += label.size(0) * label.size(1)

        total_loss = np.average(total_loss)
        accuracy = total_correct / total_samples
        
        self.model.train()

        return total_loss, accuracy


    def train(self, setting):

        # print('---------------Prepare train dataset---------------')

        _, train_loader = self._get_data(is_vmd=self.args.is_vmd, flag='train', dataset=self.args.dataset)

        print('---------------Get train dataset done---------------')

        dir_name = "_".join([f"{v}" for v in setting.values()])
        path = os.path.join(self.args.checkpoints, dir_name)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()
               
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs, attns = self.model(batch_x, padding_mask)
                loss = criterion(outputs, label.long())
                loss.backward()
                model_optim.step()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if self.args.model !='FEDformer' and self.args.model !='Pyraformer':
                attns = attns[0].cpu().detach().numpy() 
            
            print("Epoch: {} cost time: {:.4f}".format(epoch + 1, time.time() - epoch_time))

            train_loss.append(loss.item())
            train_loss = np.average(train_loss)

            formatted_setting = ' | '.join([f"{k}:{v}" for k, v in setting.items()])
            print('>>Start validation : {}>>'.format(formatted_setting))
            
            vali_loss, val_accuracy = self.vali(criterion)
            
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} | Vali Loss: {3:.3f} | Vali Acc: {4:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy))

            early_stopping(-val_accuracy, self.model, path, attns)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path), map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=1):

        # print('---------------Prepare test dataset---------------')

        _, test_loader = self._get_data(is_vmd=self.args.is_vmd, flag='test', dataset=self.args.dataset)

        print('---------------Get test dataset done!---------------')


        print('---------------Loading model------------------------')

        self.model.load_state_dict(torch.load(self.args.checkpoints_test_only))

        print('---------------Load successfully!-------------------')

        accuracy = 0

        dir_name = "_".join([f"{v}" for v in setting.values()])
        folder_path = os.path.join('test_results', dir_name)

        # folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pred_label_all = None
        self.model.eval()
        num_classes = 6

        acc_num = torch.zeros((1, num_classes))
        target_num = torch.zeros((1, num_classes))
        predict_num = torch.zeros((1, num_classes)) 

        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
            
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs, _ = self.model(batch_x, padding_mask)
                pred = outputs.detach().cpu()
                probs = torch.nn.functional.softmax(pred)  # (total_samples, num_classes) est. prob. for each class and sample
                predictions = torch.argmax(probs, dim=1) # (total_samples,) int class index for each sample
                
                # save predictions for further plot
                pred_label_all = np.concatenate(
                (pred_label_all, predictions.cpu())) if pred_label_all is not None else predictions.cpu()
                
                outputs = outputs.reshape(-1, num_classes)

                pre_mask = torch.zeros(outputs.size()).scatter_(1, predictions.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                test_label = label.long()

                tar_mask = torch.zeros(outputs.size()).scatter_(1, test_label.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)
        
        recall = acc_num / target_num
        precision = acc_num / predict_num + float('1e-8')
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
        
        print('Test Acc {}'.format(accuracy))
        print('recall {}'.format(recall))
        print('precision {}'.format(precision))
        print('F1-score {}'.format(F1))

        txt_file_name = 'cls_result.txt'

        f = open(os.path.join(folder_path, txt_file_name), 'a')
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('recall:{}'.format(recall))
        f.write('\n')
        f.write('precision:{}'.format(precision))
        f.write('\n')
        f.write('F1:{}'.format(F1))
        f.write('\n')
        f.close()
        
        npyfile_name = 'cls_result.npy'
        np.save(os.path.join(folder_path, npyfile_name), pred_label_all)
        print('---------------Predicted results saved :)-----------')
        

        return
