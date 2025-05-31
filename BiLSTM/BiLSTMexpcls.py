from test_datafactory2 import data_provider
# from test_datafactory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import LSTMEarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

torch.cuda.set_device(1)

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='train', is_vmd=self.args.is_vmd)
        vali_data, vali_loader = self._get_data(flag='val', is_vmd=self.args.is_vmd)
        test_data, test_loader = self._get_data(flag='test', is_vmd=self.args.is_vmd)

        self.args.pred_len = 0
        
        # self.args.num_class = len(np.unique(train_data.label_raw)) # obtain class number
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, is_vmd):
        data_set, data_loader = data_provider(self.args, is_vmd, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # adapt the varing learning rate
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.99)
                              
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def self_supervised(self, setting):

        _, train_loader = self._get_data(is_vmd=self.args.is_vmd , flag='self_supervised')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.L1Loss()
       
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(train_loader):
                
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                x_mark_enc_binary = x_mark_enc_binary.float().to(self.device)

                outputs, _ = self.model(batch_x)
                
                # we keep the model without the later projection, so that we can load at the supervised training stage.

                projection_layer = nn.Linear(self.args.d_model, 1).to(self.device)
                projected_outputs = projection_layer(outputs)

                loss = criterion_1(torch.squeeze(projected_outputs), torch.sum(batch_x, dim=-1)) # we change the type and squeeze
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            
            vali_mse_loss, vali_mae_loss = self.self_supervised_vali(criterion_1, criterion_2)
            val_test_mse_loss, val_test_mae_loss = self.self_supervised_vali_test(criterion_1, criterion_2)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali mse Loss: {3:.3f} Vali mae loss: {4:.3f} Val Test mse Loss: {5:.3f} Val Test mae loss: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_mse_loss, vali_mae_loss, val_test_mse_loss, val_test_mae_loss))
            early_stopping(-train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + '/' + 'ssl_checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def self_supervised_vali(self,  criterion_1, criterion_2):
        total_MSE_loss = []
        total_MAE_loss = []
        
        _, vali_loader = self._get_data(is_vmd=self.args.is_vmd, flag='val')
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                x_mark_enc_binary = x_mark_enc_binary.to(self.device)
                outputs, _ = self.model(batch_x, padding_mask, x_mark_enc_binary, None, None)
                feature_dim = 64
                projection_layer = nn.Linear(feature_dim, 1).to("cuda:1")
                projected_outputs = projection_layer(outputs)
                # loss = criterion(pred, label.long().cpu())
                MSEloss = criterion_1(torch.squeeze(projected_outputs), torch.sum(batch_x, dim=-1)) # we change the type and squeeze
                MAEloss = criterion_2(torch.squeeze(projected_outputs), torch.sum(batch_x, dim=-1))

                total_MSE_loss.append(MSEloss.item())
                total_MAE_loss.append(MAEloss.item())

        total_MSE_loss = np.average(total_MSE_loss)
        total_MAE_loss = np.average(total_MAE_loss)

        self.model.train()
        return total_MSE_loss, total_MAE_loss
    
    def self_supervised_vali_test(self,  criterion_1, criterion_2):
        total_MSE_loss = []
        total_MAE_loss = []

        _, vali_test_loader = self._get_data(is_vmd=self.args.is_vmd, flag='val_test')
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(vali_test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                x_mark_enc_binary = x_mark_enc_binary.to(self.device)
                outputs, _ = self.model(batch_x, padding_mask, x_mark_enc_binary, None, None)
                feature_dim = 64
                projection_layer = nn.Linear(feature_dim, 1).to("cuda:1")
                projected_outputs = projection_layer(outputs)

                # loss = criterion(pred, label.long().cpu())
                MSEloss = criterion_1(torch.squeeze(projected_outputs), torch.sum(batch_x, dim=-1)) # we change the type and squeeze
                MAEloss = criterion_2(torch.squeeze(projected_outputs), torch.sum(batch_x, dim=-1))

                total_MSE_loss.append(MSEloss.cpu())
                total_MAE_loss.append(MAEloss.cpu())

        total_MSE_loss = np.average(total_MSE_loss)
        total_MAE_loss = np.average(total_MAE_loss)
        self.model.train()

        return total_MSE_loss, total_MAE_loss

    def pre_train(self, setting):
        _, train_loader = self._get_data(is_vmd=self.args.is_vmd, flag='train')
       
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # best_model_path = '/home/dell/disk/Jinlong/Time-Series-Library-main/checkpoints/self_supervised_iTransformer_TSF_S_ft255_sl100_ll255_pl64_dm8_nh2_el1_dl64_df1_fctimeF_ebTrue_dttest_projection_0/checkpoint.pth'

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion = self._select_criterion()
        criterion = nn.CrossEntropyLoss()
        # self.model.load_state_dict(torch.load(best_model_path))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(train_loader):
                # all of the above are 32, 255, 1 
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                x_mark_enc_binary = x_mark_enc_binary.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs, _ = self.model(batch_x, padding_mask, x_mark_enc_binary, None, None)

                outputs = torch.swapaxes(outputs, -1, 1)
                outputs = torch.unsqueeze(outputs, dim=-1)

                loss = criterion(outputs, label.long()) # we change the type and squeeze

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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            
            vali_loss, val_accuracy = self.vali(criterion)
            val_test_loss, val_test_accuracy = self.vali_test(criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Val Test Loss: {5:.3f} Val Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_test_loss, val_test_accuracy))
            early_stopping(-train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model
    
    def vali(self,  criterion):
        total_loss = []
        accuracy = 0
        total_samples = 0
        total_correct = 0
        _, vali_loader = self._get_data(is_vmd=self.args.is_vmd, flag='val')
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(vali_loader):
            # for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                # x_mark_enc_binary = x_mark_enc_binary.to(self.device)
                batch_x = torch.transpose(batch_x, -1, 1)
                outputs = self.model(batch_x)
                outputs = outputs.reshape(-1, self.args.num_class, self.args.seq_len2, 1)
                # outputs = outputs.reshape(-1, self.args.num_class, self.args.seq_len, 1)

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

    def vali_test(self,  criterion):
        total_loss = []
        accuracy = 0
        total_samples = 0
        total_correct = 0
        _, vali_test_loader = self._get_data(is_vmd=self.args.is_vmd, flag='val_test')
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            # for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(vali_test_loader):
            for i, (batch_x, label) in enumerate(vali_test_loader):
                batch_x = batch_x.float().to(self.device)

                label = label.to(self.device)

                batch_x = torch.transpose(batch_x, -1, 1)
                outputs = self.model(batch_x)
                outputs = outputs.reshape(-1, self.args.num_class, self.args.seq_len2, 1)
                # outputs = outputs.reshape(-1, self.args.num_class, self.args.seq_len, 1)
               
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
        _, train_loader = self._get_data(is_vmd=self.args.is_vmd, flag='train')
       
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = LSTMEarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(train_loader):
            # for i, (batch_x, label) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                
                label = label.to(self.device)
                # x_mark_enc_binary = x_mark_enc_binary.to(self.device)
                batch_x = torch.transpose(batch_x, -1, 1)
                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs = self.model(batch_x) # batch_x: 16 1 255
                outputs = outputs.reshape(-1, self.args.num_class2, self.args.seq_len2, 1)
                # outputs = outputs.reshape(-1, self.args.num_class2, self.args.seq_len, 1)
                loss = criterion(outputs, label.long()) # we change the type and squeeze
              
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
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            
            vali_loss, val_accuracy = self.vali(criterion)
            # val_test_loss, val_test_accuracy = self.vali_test(criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        

        # # Load the test model
        # best_model_path = path + '/' + 'checkpoint.pth'
        # test_model = YourModelClass()
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda:1')))
        # self.model.load_state_dict(torch.load(best_model_path), map_location='cuda:1')

        return self.model


    def test(self, setting):
        _, test_loader = self._get_data(is_vmd=self.args.is_vmd, flag='test')
      
        print('loading model')

        if self.args.is_training:
             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        else:
            self.model.load_state_dict(torch.load(self.args.checkpoints_test_only))

        print('load successfully!')
    
        accuracy = 0
        total_samples = 0
        total_correct = 0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pred_label_all = None
        self.model.eval()
        num_classes = 6
        acc_num = torch.zeros((1, num_classes))
        target_num = torch.zeros((1, num_classes))
        predict_num = torch.zeros((1, num_classes)) 

        with torch.no_grad():
            start_time = time.time()  # Start time
            for i, (batch_x, label, padding_mask, x_mark_enc_binary) in enumerate(test_loader):
            # for i, (batch_x, label) in enumerate(test_loader):
            
                batch_x = batch_x.float().to(self.device)
                
                label = label.to(self.device)
                
                batch_x = torch.transpose(batch_x, -1, 1)
                outputs = self.model(batch_x)
                outputs = outputs.reshape(-1, self.args.num_class, self.args.seq_len2, 1)
                # outputs = outputs.reshape(-1, self.args.num_class, self.args.seq_len, 1)

                pred = outputs.detach().cpu()
                probs = torch.nn.functional.softmax(pred)  # (total_samples, num_classes) est. prob. for each class and sample
                predictions = torch.argmax(probs, dim=1) # (total_samples,) int class index for each sample
                total_correct += (predictions.cpu() == label.cpu()).sum().item()
                total_samples += label.size(0) * label.size(1)
                # preds.append(outputs.detach())
                # trues.append(label)
                pred_label_all = np.concatenate(
                (pred_label_all, predictions.cpu())) if pred_label_all is not None else predictions.cpu()
                
                # outputs = outputs.reshape(-1, num_classes)

                # pre_mask = torch.zeros(outputs.size()).scatter_(1, predictions.cpu().view(-1, 1), 1.)
                # predict_num += pre_mask.sum(0)
                # test_label = label.long()

                # tar_mask = torch.zeros(outputs.size()).scatter_(1, test_label.data.cpu().view(-1, 1), 1.)
                # target_num += tar_mask.sum(0)
                # acc_mask = pre_mask * tar_mask
                # acc_num += acc_mask.sum(0)


        # accuracy = total_correct / total_samples
        # # result save
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(os.path.join(folder_path, 'bilstm_nz_facies.npy'), pred_label_all)
        end_time = time.time()  # End time
        total_test_time = end_time - start_time  # Total test time
        print('Total test time: {:.2f} seconds'.format(total_test_time))


        # recall = acc_num / target_num
        # precision = acc_num / predict_num + float('1e-8')
        # F1 = 2 * recall * precision / (recall + precision)
        # accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

        # print('Predicted results saved!:)')
        # print('accuracy:{}'.format(accuracy))

        # print('Test Acc {}'.format(accuracy))
        # print('recall {}'.format(recall))
        # print('precision {}'.format(precision))
        # print('F1-score {}'.format(F1))

        # file_name='result_classification.txt'
        # f = open(os.path.join(folder_path,file_name), 'a')
        # f.write(setting + "  \n")
        # f.write('accuracy:{}'.format(accuracy))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        return
