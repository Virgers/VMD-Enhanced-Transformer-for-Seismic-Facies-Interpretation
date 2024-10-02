from data_provider.ssl_datafactory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual, selfEarlyStopping
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from thop import profile

warnings.filterwarnings('ignore')


class Exp_ssl(Exp_Basic):
    def __init__(self, args):
        super(Exp_ssl, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                
                """
                B = batch size
                T = seq len
                N = number of features
                """

                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()
                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss


    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
      
        # path = os.path.join(self.args.checkpoints, setting)

        dir_name = "_".join([f"{v}" for v in setting.values()])
        path = os.path.join(self.args.checkpoints, dir_name)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        
        self_early_stopping = selfEarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # print model parameters 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")

        # Calculate FLOPs
        sample_input = next(iter(train_loader))[0].float().to(self.device)  # Take a sample input
        sample_mask = torch.rand_like(sample_input).to(self.device)
        flops, params = profile(self.model, inputs=(sample_input, sample_mask), verbose=False)

        print(f"Total FLOPs: {flops}")

        total_epoch_duration = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained

                # using the random mask to generate inp
                # if self.args.mask_type == 'random':
                #   inp = batch_x.masked_fill(mask == 0, 0)

                # The results I found are the two masks below are not suitable
                # using composite mask 
                if self.args.mask_type == 'composite':
                    composite_mask = mask * batch_x_mark
                    inp = batch_x.masked_fill(composite_mask == 0, 0)
                
                # # using the intended mask solely
                # if self.args.mask_type == 'intended':
                #     inp = batch_x.masked_fill(batch_x_mark == 0, 0)

                outputs = self.model(inp, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            
            epoch_duration = time.time() - epoch_time
            total_epoch_duration += epoch_duration
            print(f"Epoch: {epoch + 1} cost time: {epoch_duration:.3f} seconds")

            # print(f"Epoch: {epoch + 1} cost time: {epoch_duration:.2f} seconds")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            self_early_stopping(vali_loss, self.model, path)
            if self_early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # save models
            
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        # torch.save(self.model.state_dict(), best_model_path)
        avg_epoch_duration = total_epoch_duration / self.args.train_epochs
        print(f"average epoch time: {avg_epoch_duration:.3f} seconds")

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
               
                # imputation
                outputs = self.model(inp, batch_x_mark)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 100 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        # formatted_mse = [f'{value:.5f}' for value in mse]
        # formatted_mae = [f'{value:.5f}' for value in mae]
    
        print(f'mse: {mse:.5f}, mae: {mae:.5f}')
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return
