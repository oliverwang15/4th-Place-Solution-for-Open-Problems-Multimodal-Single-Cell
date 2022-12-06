import time
import torch
import pickle
from loss import partial_correlation_score_torch_faster
from tqdm.notebook import tqdm
import copy
import os
from torch.utils import tensorboard
from DataloaderCOO import DataLoaderCOO

class Multi_Trainer:
    def __init__(self,device,config):
        self.device = device
        self.config = config
        self.time_stamp = str(time.time())
        
    def train_fn(self,model, optimizer, criterion, dl_train):
        
        loss_list = []
        model.train()
        for inpt, tgt in tqdm(dl_train,leave=False):
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
           
        avg_loss = sum(loss_list).cpu().item()/len(loss_list)
        self.logger.add_scalar("train/loss_epoch",avg_loss,self.train_epochs)
        del loss_list,inpt,tgt,loss,pred

        lr = optimizer.param_groups[0]["lr"]
        self.logger.add_scalar("val/learning_rate",lr,self.val_epochs)
        torch.cuda.empty_cache()
        return {"loss":avg_loss}

    def valid_fn(self,model, criterion, dl_valid):
        
        loss_list = []
        pred_all = []
        model.eval()
        for inpt, tgt in tqdm(dl_valid,leave=False):
            self.val_steps += 1
            mb_size = inpt.shape[0]
            inpt = inpt.to(self.device)
            tgt = tgt.to(self.device)
            with torch.no_grad():
                pred = model(inpt)
                pred_all.append(pred)
            loss = criterion(pred, tgt)
            self.logger.add_scalar("val/loss_step",loss,self.val_steps)
            loss_list.append(loss.detach())
        
        pred_all = torch.concat(pred_all,dim = 0)
        y = pred_all@self.components
        score = torch.mean(partial_correlation_score_torch_faster(self.valid_target,y.cpu()))
        avg_loss = sum(loss_list).cpu().item()/len(loss_list)
        self.logger.add_scalar("val/loss_epoch",avg_loss,self.val_epochs)
        self.logger.add_scalar("val/pearson",score,self.val_epochs)
        del loss_list,inpt,tgt,loss,pred,pred_all,y
        torch.cuda.empty_cache()
        return {"loss":avg_loss, "score":score}

    def train_model(self,model, optimizer, scheduler,dl_train, dl_valid, save_prefix):

        criterion = self.config["criterion"]
        
        save_params_filename = save_prefix+"_best_params.pth"
        save_config_filename = save_prefix+"_config.pkl"
        best_score = None

        for epoch in tqdm(range(self.config["max_epochs"]),leave=False):
            self.train_epochs += 1
            self.val_epochs += 1
            log_train = self.train_fn(model, optimizer, criterion, dl_train)
            log_valid = self.valid_fn(model, criterion, dl_valid)
            scheduler.step()
            print(f"epoch-{epoch} train_loss:{log_train['loss']} val_loss:{log_valid['loss']} corr_score:{log_valid['score']}")
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
                print("out of patience")
                break
        
        return best_score

    def train_one_fold(self,num_fold,FOLDS_LIST,train_inputs,train_targets,model,components):
        
        self.logger = tensorboard.SummaryWriter(self.config["tb_dir"]+f"{self.time_stamp}/{num_fold}/")
        self.train_steps = 0
        self.train_epochs = 0
        self.val_steps = 0
        self.val_epochs = 0

        train_idx, valid_idx = FOLDS_LIST[num_fold]
        
        train_inputs = torch.tensor(train_inputs,dtype=torch.float)
        train_targets = torch.tensor(train_targets,dtype=torch.float)

        self.components = components
        
        train_data = train_inputs[train_idx]
        valid_data = train_inputs[valid_idx]
        train_target = train_targets[train_idx]
        valid_target = train_targets[valid_idx]

        self.valid_target = valid_target

        ds_train = torch.utils.data.TensorDataset(train_data,train_target)
        ds_valid = torch.utils.data.TensorDataset(valid_data,valid_target)

        dl_train = torch.utils.data.DataLoader(ds_train,
                    batch_size=self.config["batch_size"], shuffle=True, drop_last=False)
        dl_valid = torch.utils.data.DataLoader(ds_valid, 
                    batch_size=self.config["batch_size"], shuffle=False, drop_last=False)
        
        model.to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), **self.config["optimizerparams"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config["milestones"], gamma=0.1,verbose  = True)
        best_score = self.train_model(model, optimizer, scheduler,dl_train, dl_valid, save_prefix="f%i"%num_fold)

        return best_score


class Cite_Trainer:
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

        for epoch in tqdm(range(self.config["max_epochs"])):
            self.train_epochs += 1
            self.val_epochs += 1
            log_train = self.train_fn(model, optimizer, criterion, dl_train)
            log_valid = self.valid_fn(model, criterion, dl_valid)
            self.model = model
            
            scheduler.step()
            print(f"epoch-{epoch} train_loss:{log_train['loss']} val_loss:{log_valid['loss']} corr_score:{log_valid['score']}")
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
                print("out of patience")
                break

        return best_score

    def train_one_fold(self,num_fold,FOLDS_LIST,train_inputs,train_targets,model,config):
        self.config = config
        self.logger = tensorboard.SummaryWriter(self.config["tb_dir"]+f"{self.time_stamp}_{num_fold}/")
        self.train_steps = 0
        self.train_epochs = 0
        self.val_steps = 0
        self.val_epochs = 0

        train_idx, valid_idx = FOLDS_LIST[num_fold]
        
        train_inputs = torch.tensor(train_inputs,dtype=torch.float)
        train_targets = torch.tensor(train_targets,dtype=torch.float)
        
        train_data = train_inputs[train_idx]
        valid_data = train_inputs[valid_idx]
        train_target = train_targets[train_idx]
        valid_target = train_targets[valid_idx]

        ds_train = torch.utils.data.TensorDataset(train_data,train_target)
        ds_valid = torch.utils.data.TensorDataset(valid_data,valid_target)

        dl_train = torch.utils.data.DataLoader(ds_train,
                    batch_size=self.config["batch_size"], shuffle=True, drop_last=False)
        dl_valid = torch.utils.data.DataLoader(ds_valid, 
                    batch_size=self.config["batch_size"], shuffle=False, drop_last=False)
        
        model.to(self.device)
        
        optimizer = self.config["optimizer"](model.parameters(), **self.config["optimizerparams"])
        scheduler = self.config["scheduler"](optimizer, **self.config["schedulerparams"])
      
        best_score = self.train_model(model, optimizer, scheduler,dl_train, dl_valid, save_prefix="f%i"%num_fold)

        return best_score


class Cite_Trainer_Stacking:
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

        for epoch in tqdm(range(self.config["max_epochs"])):
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
                print("out of patience")
                break

        return best_score

    def train_one_fold(self,num_fold,FOLDS_LIST,train_inputs,train_targets,model,config):
        self.config = config
        self.logger = tensorboard.SummaryWriter(self.config["tb_dir"]+f"{self.time_stamp}_{num_fold}/")
        self.train_steps = 0
        self.train_epochs = 0
        self.val_steps = 0
        self.val_epochs = 0

        train_idx, valid_idx = FOLDS_LIST[num_fold]
        
        train_inputs = torch.tensor(train_inputs,dtype=torch.float)
        train_targets = torch.tensor(train_targets,dtype=torch.float)
        
        train_data = train_inputs[train_idx]
        valid_data = train_inputs[valid_idx]
        train_target = train_targets[train_idx]
        valid_target = train_targets[valid_idx]

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
        best_score = self.train_model(model, optimizer, scheduler,dl_train, dl_valid, save_prefix=f"{self.config['model_dir']}f{num_fold}")

        return best_score


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

        for epoch in tqdm(range(self.config["max_epochs"])):
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
                print("out of patience")
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
