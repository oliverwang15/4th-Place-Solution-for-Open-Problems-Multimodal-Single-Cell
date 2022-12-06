import torch
import glob
import gc
import os
import pickle
import numpy as np
from models import NN,CNN,final_linear,MultiOutputLGBMRegressor,MultiOutputCatboostRegressor
from sklearn.multioutput import MultiOutputRegressor
from trainer import Cite_Trainer_Stacking_New

class NNWrapper(object):
    def __init__(self,params):
        self.config = params
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
    def train(self, x, y,val_x,val_y,layer = "layer_1"):
        print(f'NN Fold {layer}..')
        trainer = Cite_Trainer_Stacking_New(self.device)
        self.config["tb_dir"] = f"./log/{layer}/"
        self.config["model_dir"] = f"{layer}/torch/"
        
        model = NN(self.config)
        best_score = trainer.train_one_fold(x,y,val_x,val_y,model,self.config)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)

    def predict(self,x,layer = "layer_1"):

        # load trained models
        model_list = []
        for fn in glob.glob(f"{layer}/torch/*_best_params.pth"):
            prefix = fn[:-len("_best_params.pth")]
            config_fn = prefix + "_config.pkl"
            config = pickle.load(open(config_fn, "rb"))
            model = NN(config)
            model.to("cpu")
            params = torch.load(fn)
            model.load_state_dict(params)
            model_list.append(model)


        # load data
        len_ = x.shape[0]
        x = torch.tensor(x,dtype=torch.float)
        x = torch.utils.data.TensorDataset(x)
        x = torch.utils.data.DataLoader(x, batch_size=4096, shuffle=False, drop_last=False)

        # start predicting
        pred_res = np.zeros((len_, self.config["output_num"]))
        cur = 0
        for inpt in x:
            inpt = inpt[0]
            with torch.no_grad():
                pred_list = []
                inpt = inpt.to(self.device)
                for id,model in enumerate(model_list):
                    model.to(self.device)
                    model.eval()
                    pred = model(inpt)
                    model.to("cpu")
                    # if layer == "layer_3":
                    if 1:
                        pred = self.std(pred.cpu().numpy())
                    else:
                        pred = pred.cpu().numpy()
                    pred_list.append(pred)

                pred = sum(pred_list)/len(pred_list)
            
            pred_res[cur:cur+pred.shape[0],:] += pred
            cur += pred.shape[0]
        del model_list
        gc.collect()
        torch.cuda.empty_cache()
        return pred_res

class CNNWrapper(object):
    def __init__(self,params):
        self.config = params
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
    def train(self, x, y,val_x,val_y,layer = "layer_1"):
        print(f'CNN Fold {layer}..')
        trainer = Cite_Trainer_Stacking_New(self.device)
        self.config["tb_dir"] = f"./log/{layer}/"
        self.config["model_dir"] = f"{layer}/cnn/"
        
        model = CNN(self.config)
        best_score = trainer.train_one_fold(x,y,val_x,val_y,model,self.config)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)

    def predict(self,x,layer = "layer_1"):
        # load trained models
        model_list = []
        for fn in glob.glob(f"{layer}/cnn/*_best_params.pth"):
            prefix = fn[:-len("_best_params.pth")]
            config_fn = prefix + "_config.pkl"
            config = pickle.load(open(config_fn, "rb"))
            model = CNN(config)
            model.to("cpu")
            params = torch.load(fn)
            model.load_state_dict(params)
            model_list.append(model)

        # load data
        len_ = x.shape[0]
        x = torch.tensor(x,dtype=torch.float)
        x = torch.utils.data.TensorDataset(x)
        x = torch.utils.data.DataLoader(x, batch_size=4096, shuffle=False, drop_last=False)

        # start predicting
        pred_res = np.zeros((len_, self.config["output_num"]))
        cur = 0
        for inpt in x:
            inpt = inpt[0]
            with torch.no_grad():
                pred_list = []
                inpt = inpt.to(self.device)
                for id,model in enumerate(model_list):
                    model.to(self.device)
                    model.eval()
                    pred = model(inpt)
                    model.to("cpu")
                    # if layer == "layer_3":
                    if 1:
                        pred = self.std(pred.cpu().numpy())
                    else:
                        pred = pred.cpu().numpy()
                    pred_list.append(pred)

                pred = sum(pred_list)/len(pred_list)
            
            pred_res[cur:cur+pred.shape[0],:] += pred
            cur += pred.shape[0]
        del model_list
        gc.collect()
        torch.cuda.empty_cache()
        return pred_res

class FinalWrapper(object):
    def __init__(self,params):
        self.config = params
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
    def train(self, x, y,val_x,val_y,layer = "layer_1"):
        print(f'Final MLP Fold {layer}..')
        trainer = Cite_Trainer_Stacking_New(self.device)
        self.config["tb_dir"] = f"./log/{layer}/"
        self.config["model_dir"] = f"{layer}/final/"

        model = final_linear(self.config)
        best_score = trainer.train_one_fold(x,y,val_x,val_y,model,self.config)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)

    def predict(self,x,layer = "layer_1"):
        # load trained models
        model_list = []
        for fn in glob.glob(f"{layer}/final/*_best_params.pth"):
            prefix = fn[:-len("_best_params.pth")]
            config_fn = prefix + "_config.pkl"
            config = pickle.load(open(config_fn, "rb"))
            model = final_linear(config)
            model.to("cpu")
            params = torch.load(fn)
            model.load_state_dict(params)
            model_list.append(model)


        # load data
        len_ = x.shape[0]
        x = torch.tensor(x,dtype=torch.float)
        x = torch.utils.data.TensorDataset(x)
        x = torch.utils.data.DataLoader(x, batch_size=4096, shuffle=False, drop_last=False)

        # start predicting
        pred_res = np.zeros((len_, self.config["output_num"]))
        cur = 0
        for inpt in x:
            inpt = inpt[0]
            with torch.no_grad():
                pred_list = []
                inpt = inpt.to(self.device)
                for id,model in enumerate(model_list):
                    model.to(self.device)
                    model.eval()
                    pred = model(inpt)
                    model.to("cpu")
                    if layer == "layer_3":
                        pred = self.std(pred.cpu().numpy())
                    else:
                        pred = pred.cpu().numpy()
                    pred_list.append(pred)

                pred = sum(pred_list)/len(pred_list)
            
            pred_res[cur:cur+pred.shape[0],:] += pred
            cur += pred.shape[0]
        del model_list
        gc.collect()
        torch.cuda.empty_cache()
        return pred_res

class CatboostWrapper(object):
    def __init__(self, seed=0, params=None):
        params['random_state'] = seed
        self.params = params

    def train(self, x, y,val_x,val_y,layer = "layer_1"):
        self.params["task_type"] ="GPU"

        Xtr, Xva = x,val_x
        Ytr, Yva = y,val_y
        print(f'Catboost Fold {layer}..')
        model = MultiOutputCatboostRegressor(self.params)
        model.fit(Xtr, Ytr,Xva,Yva,)

        del Xtr, Ytr
        del Xva, Yva
        gc.collect()
        
        d_path = f"{layer}/catboost/Fold/"
        os.makedirs(d_path,exist_ok=True)
        model.dump(d_path)

    def predict(self, x,layer = "layer_1"):
        self.params["task_type"] ="CPU"
        model_path = f"{layer}/catboost/Fol*"
        model_list = glob.glob(model_path)
        preds = np.zeros((x.shape[0], 140))
        for id,fn in enumerate(model_list):
            model_ = MultiOutputCatboostRegressor(self.params)
            model_.load(fn+"/")
            preds += model_.predict(x)
            gc.collect()
        preds /= len(model_list)
        return preds

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)

class LightGBMWrapper(object):
    def __init__(self, seed=0, params=None):
        params['seed'] = seed
        self.params = params

    def train(self, x, y,val_x,val_y,layer = "layer_1"):
        Xtr, Xva = x,val_x
        Ytr, Yva = y,val_y
        print(f'LGBM Fold {layer}..')
        model = MultiOutputLGBMRegressor(self.params)
        model.fit(Xtr, Ytr,Xva,Yva,)
        
        del Xtr, Ytr,Xva,Yva
        gc.collect()
        d_path = f"{layer}/lgbm/Fold/"
        os.makedirs(d_path,exist_ok=True)
        model.dump(d_path)

    def predict(self, x,layer = "layer_1"):
        model_path = f"{layer}/lgbm/Fol*"
        model_list = glob.glob(model_path)
        preds = np.zeros((x.shape[0], 140))
        for id,fn in enumerate(model_list):
            model_ = MultiOutputLGBMRegressor(self.params)
            model_.load(fn+"/")
            preds += model_.predict(x)
            gc.collect()
        preds /= len(model_list)
        return preds

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)

class SklearnWrapper(object):
    def __init__(self,seed=0, params=None):
        self.params = params

    def train(self,clf, x, y,val_x,val_y,layer = "layer_1"):
        Xtr, Xva = x,val_x
        Ytr, Yva = y,val_y
        model = clf(**self.params)
        model_type = str(type(model)).split("'")[1].split(".")[-1]
        print(f'{model_type} Fold {layer}..')
        model.fit(Xtr, Ytr)

        del Xtr, Ytr
        del Xva, Yva
        gc.collect()
        
        d_path = f'{layer}/{model_type}/'
        os.makedirs(d_path,exist_ok=True)
        with open(d_path+f"Fold.pkl","wb") as f:
            pickle.dump(model,f)

  
    def predict(self,clf, x,layer = "layer_1"):
        model = clf(**self.params)
        model_type = str(type(model)).split("'")[1].split(".")[-1]
        model_path = f"{layer}/{model_type}/Fol*.pkl"
        model_list = glob.glob(model_path)

        preds = np.zeros((x.shape[0], 140))
        for id,fn in enumerate(model_list):
            with open(fn,"rb") as f:
                model = pickle.load(f)
            preds += model.predict(x)
            gc.collect()
        preds /= len(model_list)
        return preds

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)

class MultiOutputSklearnWrapper(SklearnWrapper):
    def train(self,clf, x, y,val_x,val_y,layer = "layer_1"):
        Xtr, Xva = x,val_x
        Ytr, Yva = y,val_y
        model = MultiOutputRegressor(clf(**self.params),n_jobs=19)
        model_type = str(type(model.estimator)).split("'")[1].split(".")[-1]
        print(f'{model_type} Fold {layer}..')
        model.fit(Xtr, Ytr)

        del Xtr, Ytr
        del Xva, Yva
        gc.collect()
        d_path = f'{layer}/{model_type}/'
        os.makedirs(d_path,exist_ok=True)
        with open(d_path+f"Fold.pkl","wb") as f:
            pickle.dump(model,f)
