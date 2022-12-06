import os
import torch
import time
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils import tensorboard

class Cite_Trainer_Stacking_New:
    def __init__(self,device):
        self.device = device
        self.time_stamp = str(time.time())
        
    def train_fn(self,model, optimizer, criterion, dl_train):
        loss_list = []
        all_mae = []
        all_mse = []
        model.train()
        for inpt, tgt in dl_train:
            self.train_steps +=1
            mb_size = inpt.shape[0]

            optimizer.zero_grad()
            inpt = inpt.to(self.device)
            tgt = tgt.to(self.device)
            pred = model(inpt)

            loss = criterion(pred, tgt)
            loss_list.append(loss.detach())
            self.logger.add_scalar("train/loss_step",loss.detach(),self.train_steps)
            loss.backward()
            optimizer.step()
            mae = torch.nn.functional.l1_loss(pred,tgt)
            mse = torch.nn.functional.mse_loss(pred,tgt)
            self.logger.add_scalar("train/mae_step",mae,self.train_steps)
            self.logger.add_scalar("train/mse_step",mse,self.train_steps)
            all_mae.append(mae.detach())
            all_mse.append(mse.detach())

        avg_loss = sum(loss_list).cpu().item()/len(loss_list)
        mae_score = sum(all_mae).cpu().item()/len(all_mae) 
        mse_score = sum(all_mse).cpu().item()/len(all_mse) 
        self.logger.add_scalar("train/loss_epoch",avg_loss,self.train_epochs)
        self.logger.add_scalar("train/mae",mae_score,self.train_epochs)
        self.logger.add_scalar("train/mse",mse_score,self.train_epochs)

        lr = optimizer.param_groups[0]["lr"]
        self.logger.add_scalar("val/learning_rate",lr,self.val_epochs)

        return {"loss":avg_loss}

    def valid_fn(self,model, criterion, dl_valid):
        loss_list = []
        all_mae = []
        all_mse = []
        partial_correlation_scores = []
        model.eval()
        for inpt, tgt in dl_valid:
            self.val_steps += 1
            mb_size = inpt.shape[0]
            inpt = inpt.to(self.device)
            tgt = tgt.to(self.device)
            with torch.no_grad():
                pred = model(inpt)
            loss = criterion(pred, tgt)
            mae = torch.nn.functional.l1_loss(pred,tgt)
            mse = torch.nn.functional.mse_loss(pred,tgt)
            self.logger.add_scalar("val/loss_step",loss,self.val_steps)

            self.logger.add_scalar("val/mae_step",mae,self.val_steps)
            self.logger.add_scalar("val/mse_step",mse,self.val_steps)
            all_mae.append(mae.detach())
            all_mse.append(mse.detach())
            loss_list.append(loss.detach())
            partial_correlation_scores.append(partial_correlation_score_torch_faster(tgt, pred))

        avg_loss = sum(loss_list).cpu().item()/len(loss_list)
        partial_correlation_scores = torch.cat(partial_correlation_scores)
        score = torch.sum(partial_correlation_scores).cpu().item()/len(partial_correlation_scores) #correlation_score_torch(all_tgts, all_preds)
        mae_score = sum(all_mae).cpu().item()/len(all_mae) 
        mse_score = sum(all_mse).cpu().item()/len(all_mse) 
        self.logger.add_scalar("val/loss_epoch",avg_loss,self.val_epochs)
        self.logger.add_scalar("val/pearson",score,self.val_epochs)
        self.logger.add_scalar("val/mae",mae_score,self.val_epochs)
        self.logger.add_scalar("val/mse",mse_score,self.val_epochs)

        return {"loss":avg_loss, "score":score}

    def train_model(self,model, optimizer, scheduler,dl_train, dl_valid, save_prefix):

        criterion = self.config["criterion"]
        
        save_params_filename = save_prefix+"_best_params.pth"
        save_config_filename = save_prefix+"_config.pkl"
        best_score = None

        for epoch in tqdm(range(self.config["max_epochs"]),leave = False):
            self.train_epochs += 1
            self.val_epochs += 1
            log_train = self.train_fn(model, optimizer, criterion, dl_train)
            log_valid = self.valid_fn(model, criterion, dl_valid)
            self.model = model
            
            scheduler.step()
            score = log_valid["score"]
            if best_score is None or score > best_score:
                best_score = score
                patience = self.config["patience"]
                best_params = copy.deepcopy(model.state_dict())
                torch.save(best_params, save_params_filename)
                with open(save_config_filename, "wb+") as f:
                    pickle.dump(self.config,f)      
            else:
                patience -= 1
            
            if patience < 0 and self.train_epochs > self.config["min_epoch"]:
                break

        return best_score

    def train_one_fold(self,train_inputs,train_targets,val,val_targets,model,config):
        self.config = config
        self.logger = tensorboard.SummaryWriter(self.config["tb_dir"]+f"{self.time_stamp}/")
        self.train_steps = 0
        self.train_epochs = 0
        self.val_steps = 0
        self.val_epochs = 0
        
        train_data = train_inputs
        valid_data = val
        train_target = train_targets
        valid_target = val_targets

        train_data = torch.tensor(train_data,dtype=torch.float)
        valid_data = torch.tensor(valid_data,dtype=torch.float)
        train_target = torch.tensor(train_target,dtype=torch.float)
        valid_target = torch.tensor(valid_target,dtype=torch.float)

        ds_train = torch.utils.data.TensorDataset(train_data,train_target)
        ds_valid = torch.utils.data.TensorDataset(valid_data,valid_target)

        dl_train = torch.utils.data.DataLoader(ds_train,
                    batch_size=self.config["batch_size"], shuffle=True, drop_last=False)
        dl_valid = torch.utils.data.DataLoader(ds_valid, 
                    batch_size=self.config["batch_size"], shuffle=False, drop_last=False)
        
        model.to(self.device)
        
        optimizer = self.config["optimizer"](model.parameters(), **self.config["optimizerparams"])
        scheduler = self.config["scheduler"](optimizer, **self.config["schedulerparams"])
    
        os.makedirs(self.config['model_dir'],exist_ok=True)
        best_score = self.train_model(model, optimizer, scheduler,dl_train, dl_valid, save_prefix=f"{self.config['model_dir']}f{0}")

        return best_score


def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    if type(y_true) == np.ndarray: y_true = torch.tensor(y_true)
    if type(y_pred) == np.ndarray: y_pred = torch.tensor(y_pred)

    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation.
    """
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape: raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)