from trainer import correl_loss,correlation_score
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from wappers import NNWrapper,CNNWrapper,FinalWrapper,CatboostWrapper,LightGBMWrapper,SklearnWrapper,MultiOutputSklearnWrapper

class ModelStacking():
    def __init__(self, train_target, test, meta_train,config,fold_list):
        self.train_target = train_target #训练集
        self.test = test # test data
        self.meta_train = meta_train
        self.config = config 
        self.fold_list = fold_list
        self.score_record = {"layer_1":{
            "0":{},
            "1":{},
            "2":{},
        },
                            "layer_2":{
            "0":{},
            "1":{},
            "2":{},
                            },
                            "layer_3":{
            "0":{},
            "1":{},
            "2":{},
                            }}
        
        self.params_list() #初始化各模型的参数

    def params_list(self):
        
        first_torch_params = dict(
            atte_dims = 128,
            output_num = len(self.config["predict_label"]),
            input_num = self.test.shape[1],
            dropout = 0.1,
            process_dim = 768,
            
            patience = 5,
            max_epochs = 100,
            criterion = correl_loss,
            batch_size = 128,

            n_folds = 3,
            folds_to_train = [0,1,2],

            tb_dir = "./log/torch/",
            model_dir = "./models_res/layer_1/torch/",

            optimizer = torch.optim.AdamW,
            optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
            
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler = torch.optim.lr_scheduler.MultiStepLR,

            # schedulerparams = dict(mode="min", factor=0.9,patience = 4,verbose  = True,min_lr = 1e-7),
            schedulerparams = dict(milestones=[6,10,15,20,25,30], gamma=0.1,verbose  = False),
            # milestones = [30], #10,15,20,25,30  # 15,20,25,30
            min_epoch = 11,
        )

        second_torch_params = dict(
            atte_dims = 128,
            output_num = len(self.config["predict_label"]),
            input_num = len(self.config["predict_label"])*len(self.config["layer_1_model_list"]) +self.test.shape[1], # 
            dropout = 0.1,
            
            process_dim = 768,
            patience = 5,
            max_epochs = 100,
            criterion = correl_loss,
            batch_size = 128,

            n_folds = 3,
            folds_to_train = [0,1,2],

            tb_dir = "./log/torch/",
            model_dir = "./models_res/layer_1/torch/",

            optimizer = torch.optim.AdamW,
            optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
            
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler = torch.optim.lr_scheduler.MultiStepLR,

            # schedulerparams = dict(mode="min", factor=0.9,patience = 4,verbose  = True,min_lr = 1e-7),
            schedulerparams = dict(milestones=[6,10,15,20,25,30], gamma=0.1,verbose  = False),
            # milestones = [30], #10,15,20,25,30  # 15,20,25,30
            min_epoch = 11,
        )

        first_cnn_params = dict(
            atte_dims = 128,
            output_num = len(self.config["predict_label"]),
            input_num = self.test.shape[1],
            dropout = 0.1,
            
            layers = 5,
            patience = 5,
            max_epochs = 100,
            criterion = correl_loss,
            batch_size = 128,
            mlp_dims = 5,

            n_folds = 3,
            folds_to_train = [0,1,2],

            tb_dir = "./log/cnn/",
            model_dir = "./models_res/layer_1/torch/",

            optimizer = torch.optim.AdamW,
            optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
            
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler = torch.optim.lr_scheduler.MultiStepLR,

            # schedulerparams = dict(mode="min", factor=0.9,patience = 4,verbose  = True,min_lr = 1e-7),
            schedulerparams = dict(milestones=[12,15,20,25,30], gamma=0.1,verbose  = False),
            # milestones = [30], #10,15,20,25,30  # 15,20,25,30
            min_epoch = 11,
        )

        second_cnn_params = dict(
            atte_dims = 128,
            output_num = len(self.config["predict_label"]),
            input_num = len(self.config["predict_label"])*len(self.config["layer_1_model_list"]) +self.test.shape[1], # 
            dropout = 0.1,
            
            layers = 5,
            patience = 5,
            max_epochs = 100,
            criterion = correl_loss,
            batch_size = 128,
            mlp_dims = 5,

            n_folds = 3,
            folds_to_train = [0,1,2],

            tb_dir = "./log/cnn/",
            model_dir = "./models_res/layer_1/torch/",

            optimizer = torch.optim.AdamW,
            optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
            
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler = torch.optim.lr_scheduler.MultiStepLR,

            # schedulerparams = dict(mode="min", factor=0.9,patience = 4,verbose  = True,min_lr = 1e-7),
            schedulerparams = dict(milestones=[12,15,20,25,30], gamma=0.1,verbose  = False),
            # milestones = [30], #10,15,20,25,30  # 15,20,25,30
            min_epoch = 11,
        )

        knn_params = {
            "n_neighbors":20,
        }

        mlp_params = {
            "early_stopping":True
        }

        svr_params = {
            "C":1.9,
            "epsilon":0.00,
        }

        en_params = {
            "random_state":42
        }

        ridge_params = {
            "random_state":42,
        }

        kernel = RBF(length_scale = 20)
        kr_params = {
            "alpha":1.7,
            "kernel":kernel
        }

        rf_params = {
            'n_jobs': 19,
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_leaf': 2,
            "verbose":0,
        }
        
        et_params = {
            'n_jobs': 19,
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_leaf': 2,
            "verbose":0,
        }
        
        cb_params = {
            'learning_rate': 0.1, 
            'depth': 8, 
            'l2_leaf_reg': 4, 
            'loss_function': 'RMSE', 
            'eval_metric': 'RMSE', 
            'task_type': 'GPU', 
            'iterations': 10000,
            'od_type': 'Iter', 
            'boosting_type': 'Plain', 
            'bootstrap_type': 'Bayesian', 
            'allow_const_label': True, 
            'random_state': 1,
            }
        
        lgb_params = {
            'learning_rate': 0.1, 
            'objective': 'mse', 
            'metric': ['mse', 'mae'], 
            'n_estimators': 10000, 
            'learning_rate': 0.011322411312518462, 
            'num_leaves': 350, 
            'verbose': -1, 
            'boosting_type': 'gbdt', 
            'reg_alpha': 0.40300033428422216, 
            'reg_lambda': 1.6473388122802188, 
            'colsample_bytree': 0.5, 
            'subsample': 0.7, 
            'max_depth': -1, 
            'min_child_samples': 54, 
            'cat_smooth': 41.24648150772993,
            'device':"gpu",
            "gpu_device_id":0,
            "gpu_platform_id":1,
            }
        
        final_params = dict(
            atte_dims = 128,
            output_num = len(self.config["predict_label"]),
            input_num = len(self.config["predict_label"])* len(self.config["layer_2_model_list"]),
            model_num = len(self.config["layer_2_model_list"]),
            dropout = 0.1,
            
            process_dim = 512,
            patience = 5,
            max_epochs = 100,
            criterion = correl_loss,
            batch_size = 256,

            n_folds = 3,
            folds_to_train = [0,1,2],

            tb_dir = "./log/final/",
            model_dir = "./models_res/layer_1/torch/",

            optimizer = torch.optim.AdamW,
            optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
            
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler = torch.optim.lr_scheduler.MultiStepLR,

            # schedulerparams = dict(mode="min", factor=0.9,patience = 4,verbose  = True,min_lr = 1e-7),
            schedulerparams = dict(milestones=[9,12,15,20,25,30], gamma=0.1,verbose  = False),
            # milestones = [30], #10,15,20,25,30  # 15,20,25,30
            min_epoch = 11,
        )

        self.params_dic = {}
        self.params_dic['first_torch_params'] = first_torch_params  
        self.params_dic['second_torch_params'] = second_torch_params 
        self.params_dic['first_cnn_params'] = first_cnn_params  
        self.params_dic['second_cnn_params'] = second_cnn_params  
        self.params_dic['rf_params'] = rf_params
        # self.params_dic['xgb_params'] = xgb_params
        self.params_dic['cb_params'] = cb_params
        self.params_dic['lgb_params'] = lgb_params
        self.params_dic['final_params'] = final_params
        self.params_dic["et_params"] = et_params
        self.params_dic["ridge_params"] = ridge_params
        self.params_dic["knn_params"] = knn_params
        self.params_dic["mlp_params"] = mlp_params
        self.params_dic["svr_params"] = svr_params
        self.params_dic["en_params"] = en_params
        self.params_dic["kr_params"] = kr_params

    def fit_single_model(self,clf,train,labels,val,val_labels,next,layer_id,infer = False):
        if clf == 'torch':
            if "layer_2" in layer_id:
                model = NNWrapper(params=self.params_dic['second_torch_params'])
            else:
                model = NNWrapper(params=self.params_dic['first_torch_params'])

            if not infer:
                model.train(train,labels,val,val_labels,layer_id)
            oof_res = model.predict(val,layer_id)
            next_res = model.predict(next,layer_id)
            next_res = self.std(next_res)

        elif clf == 'CNN':
            if "layer_2" in layer_id:
                model = CNNWrapper(params=self.params_dic['second_cnn_params'])
            else:
                model = CNNWrapper(params=self.params_dic['first_cnn_params'])
            
            if not infer:
                model.train(train,labels,val,val_labels,layer_id)
            oof_res = model.predict(val,layer_id)
            next_res = model.predict(next,layer_id)
            next_res = self.std(next_res)

        elif clf == "lgbm":
            model = LightGBMWrapper(params=self.params_dic['lgb_params'])
            if not infer:
                model.train(train,labels,val,val_labels,layer_id)
            oof_res = model.predict(val,layer_id)
            next_res = model.predict(next,layer_id)

        elif clf == "catboost":
            model = CatboostWrapper(params=self.params_dic['cb_params'])
            if not infer:
                model.train(train,labels,val,val_labels,layer_id)
            oof_res = model.predict(val,layer_id)
            next_res = model.predict(next,layer_id)

        elif clf == "rf":
            model = SklearnWrapper(params=self.params_dic['rf_params'])
            if not infer:
                model.train(RandomForestRegressor,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(RandomForestRegressor,val,layer_id)
            next_res = model.predict(RandomForestRegressor,next,layer_id)

        elif clf == "et":
            model = SklearnWrapper(params=self.params_dic['rf_params'])
            if not infer:
                model.train(ExtraTreesRegressor,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(ExtraTreesRegressor,val,layer_id)
            next_res = model.predict(ExtraTreesRegressor,next,layer_id)

        elif clf == "ridge":
            model = SklearnWrapper(params=self.params_dic['ridge_params'])
            if not infer:
                model.train(Ridge,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(Ridge,val,layer_id)
            next_res = model.predict(Ridge,next,layer_id)

        elif clf == "KNN":
            model = SklearnWrapper(params=self.params_dic['knn_params'])
            if not infer:
                model.train(KNeighborsRegressor,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(KNeighborsRegressor,val,layer_id)
            next_res = model.predict(KNeighborsRegressor,next,layer_id)
         
        elif clf == "MLP":
            model = SklearnWrapper(params=self.params_dic['mlp_params'])
            if not infer:
                model.train(MLPRegressor,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(MLPRegressor,val,layer_id)
            next_res = model.predict(MLPRegressor,next,layer_id)
        
        elif clf == "SVR":
            model = MultiOutputSklearnWrapper(params=self.params_dic['svr_params'])
            if not infer:
                model.train(SVR,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(SVR,val,layer_id)
            next_res = model.predict(SVR,next,layer_id)

        elif clf == "ElasticNet":
            model = SklearnWrapper(params=self.params_dic['en_params'])
            if not infer:
                model.train(ElasticNet,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(ElasticNet,val,layer_id)
            next_res = model.predict(ElasticNet,next,layer_id)

        elif clf == "KernelRidge":
            model = SklearnWrapper(params=self.params_dic['kr_params'])
            if not infer:
                model.train(KernelRidge,train,labels,val,val_labels,layer_id)
            oof_res = model.predict(KernelRidge,val,layer_id)
            next_res = model.predict(KernelRidge,next,layer_id)

        corr_score = correlation_score(val_labels,oof_res)

        if "layer_1" in layer_id:
            tmp_dict = self.score_record["layer_1"]
            temp_fold = tmp_dict[layer_id[-1]]
            temp_fold[clf] = corr_score
            tmp_dict[layer_id[-1]] = temp_fold
            self.score_record["layer_1"] = tmp_dict
        elif "layer_2" in layer_id:
            tmp_dict = self.score_record["layer_2"]
            temp_fold = tmp_dict[layer_id[-1]]
            temp_fold[clf] = corr_score
            tmp_dict[layer_id[-1]] = temp_fold
            self.score_record["layer_2"] = tmp_dict

        print(f'{clf} {layer_id} CV score: {corr_score}')
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return next_res

    def train_first_layer(self,clf,id,infer = False):
        train = self.train_first_layer_fold
        labels = self.train_first_layer_fold_label
        val = self.valid_first_layer_fold
        val_labels = self.valid_first_layer_fold_label
        next = self.train_target[self.config["train_features"]].values

        layer_id = f"{self.surfix}layer_1/{id}"

        next_res = self.fit_single_model(clf,train,labels,val,val_labels,next,layer_id,infer)
        
        return next_res
   
    def fit_first_layer(self,infer = False):

        fold_list = self.fold_list
        
        for id,((idx_tr, idx_va),idx_test) in enumerate(fold_list):
            
            self.train_first_layer_fold = self.train_target[self.config["train_features"]].values[idx_tr]
            self.valid_first_layer_fold = self.train_target[self.config["train_features"]].values[idx_va]
            self.train_first_layer_fold_label = self.train_target[self.config["predict_label"]].values[idx_tr]
            self.valid_first_layer_fold_label = self.train_target[self.config["predict_label"]].values[idx_va]

            self.first_models_oof_return = {}
            for model in self.config["layer_1_model_list"]:
                self.first_models_oof_return[model] = self.train_first_layer(model,id,infer)
            
            second_layer_input_fold = self.train_target[self.config["train_features"]].values
            for idi,(model_name,oof_res) in enumerate(self.first_models_oof_return.items()):
                second_layer_input_fold = np.concatenate([second_layer_input_fold,oof_res],axis=1)
            
            preds = second_layer_input_fold[idx_test]
            preds = np.concatenate([preds,np.array(idx_test).reshape(-1,1)],axis = -1)
            if id == 0:
                second_layer_input = preds
            else:
                second_layer_input = np.concatenate([second_layer_input,preds],axis = 0)

        second_layer_input = pd.DataFrame(second_layer_input)
        second_layer_input = second_layer_input.set_index(second_layer_input.shape[1]-1).sort_index()
        self.second_layer_input = second_layer_input.values
        self.second_layer_input_ = second_layer_input

        gc.collect()
        torch.cuda.empty_cache()
        print("Traing of first layer finished")
                 
    def train_second_layer(self,clf,id,infer = False):
        train = self.train_second_layer_fold
        labels = self.train_second_layer_fold_label
        val = self.valid_second_layer_fold
        val_labels = self.valid_second_layer_fold_label
        next = self.second_layer_input

        layer_id = f"{self.surfix}layer_2/{id}"
        next_res = self.fit_single_model(clf,train,labels,val,val_labels,next,layer_id,infer)

        return next_res

    def fit_second_layer(self,infer = False):
        fold_list = self.fold_list

        for id,((idx_tr, idx_va),idx_test) in enumerate(fold_list):

            self.train_second_layer_fold = self.second_layer_input[idx_tr]
            self.valid_second_layer_fold = self.second_layer_input[idx_va]
            self.train_second_layer_fold_label = self.train_target[self.config["predict_label"]].values[idx_tr]
            self.valid_second_layer_fold_label = self.train_target[self.config["predict_label"]].values[idx_va]

            self.second_models_oof_return = {}
            for model in self.config["layer_2_model_list"]:
                self.second_models_oof_return[model] = self.train_second_layer(model,id,infer)
            
            for idi,(model_name,oof_res) in enumerate(self.second_models_oof_return.items()):
                if idi == 0:
                    third_layer_input_fold = oof_res
                else:
                    third_layer_input_fold = np.concatenate([third_layer_input_fold,oof_res],axis=1)
            
            preds = third_layer_input_fold[idx_test]
            preds = np.concatenate([preds,np.array(idx_test).reshape(-1,1)],axis = -1)
            if id == 0:
                third_layer_input = preds
            else:
                third_layer_input = np.concatenate([third_layer_input,preds],axis = 0)

        third_layer_input = pd.DataFrame(third_layer_input)
        third_layer_input = third_layer_input.set_index(third_layer_input.shape[1]-1).sort_index()
        self.third_layer_input = third_layer_input.values

        gc.collect()
        torch.cuda.empty_cache()
        print("Traing of second layer finished")

    def train_third_layer(self):
        last_model = self.config["layer_3_model"]
        if last_model == "mlp":
            self.final_models = FinalWrapper(params=self.params_dic['final_params'])
        elif last_model == "torch":
            self.final_models = NNWrapper(params=self.params_dic['final_params'])
        elif last_model == "catboost":
            self.final_models = CatboostWrapper(params=self.params_dic['cb_params']) 

        fold_list = self.fold_list
        
        for id,((idx_tr, idx_va),idx_test) in enumerate(fold_list):
            layer_id = f"{self.surfix}layer_3/{id}"
            train = self.third_layer_input[idx_tr]
            labels = self.train_target[self.config["predict_label"]].values[idx_tr]
            val = self.third_layer_input[idx_va]
            val_labels = self.train_target[self.config["predict_label"]].values[idx_va]

            self.final_models.train(train,labels,val,val_labels,layer_id)
            oof_res = self.final_models.predict(val,layer_id)
            corr_score = correlation_score(val_labels,oof_res)
            tmp_dict = self.score_record["layer_3"][str(id)]
            tmp_dict[last_model] = corr_score
            self.score_record["layer_3"][str(id)] = tmp_dict

        print("Traing of Third layer finished")

    def fit(self,dir_n = 0):
        self.surfix = f"./results/{dir_n}/"
        self.fit_first_layer()
        print(f"Shape of Second layer input: {self.second_layer_input.shape}")
        self.fit_second_layer()
        print(f"Shape of Third layer input: {self.third_layer_input.shape}")
        self.train_third_layer()
        print("all done!")

    def fit_last(self,dir_n = 0):
        self.surfix = f"./results/{dir_n}/"
        self.fit_first_layer(infer = True)
        print(f"Shape of Second layer input: {self.second_layer_input.shape}")
        self.fit_second_layer(infer = True)
        print(f"Shape of Third layer input: {self.third_layer_input.shape}")
        self.train_third_layer()
        print("all done!")

    def fit_second(self,dir_n = 0):
        self.surfix = f"./results/{dir_n}/"
        self.fit_first_layer(infer = True)
        print(f"Shape of Second layer input: {self.second_layer_input.shape}")
        self.fit_second_layer(infer = False)
        print(f"Shape of Third layer input: {self.third_layer_input.shape}")
        self.train_third_layer()
        print("all done!")

    def pred_single_model(self,clf,next,layer_id):
        # ["KNN","MLP","KernelRidge","ElasticNet",'ridge',"rf","et","catboost","torch"]
        if clf == 'torch':
            if "layer_2" in layer_id:
                model = NNWrapper(params=self.params_dic['second_torch_params'])
            else:
                model = NNWrapper(params=self.params_dic['first_torch_params'])


            next_res = model.predict(next,layer_id)
            next_res = self.std(next_res)

        if clf == 'CNN':
            if "layer_2" in layer_id:
                model = CNNWrapper(params=self.params_dic['second_cnn_params'])
            else:
                model = CNNWrapper(params=self.params_dic['first_cnn_params'])
            
            next_res = model.predict(next,layer_id)
            next_res = self.std(next_res)

        elif clf == "lgbm":
            model = LightGBMWrapper(params=self.params_dic['lgb_params'])
            next_res = model.predict(next,layer_id)

        elif clf == "catboost":
            model = CatboostWrapper(params=self.params_dic['cb_params'])
            next_res = model.predict(next,layer_id)

        elif clf == "rf":
            model = SklearnWrapper(params=self.params_dic['rf_params'])
            next_res = model.predict(RandomForestRegressor,next,layer_id)

        elif clf == "et":
            model = SklearnWrapper(params=self.params_dic['rf_params'])
            next_res = model.predict(ExtraTreesRegressor,next,layer_id)

        elif clf == "ridge":
            model = SklearnWrapper(params=self.params_dic['ridge_params'])
            next_res = model.predict(Ridge,next,layer_id)

        elif clf == "KNN":
            model = SklearnWrapper(params=self.params_dic['knn_params'])
            next_res = model.predict(KNeighborsRegressor,next,layer_id)
         
        elif clf == "MLP":
            model = SklearnWrapper(params=self.params_dic['mlp_params'])
            next_res = model.predict(MLPRegressor,next,layer_id)
        

        elif clf == "ElasticNet":
            model = SklearnWrapper(params=self.params_dic['en_params'])
            next_res = model.predict(ElasticNet,next,layer_id)

        elif clf == "KernelRidge":
            model = SklearnWrapper(params=self.params_dic['kr_params'])
            next_res = model.predict(KernelRidge,next,layer_id)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return next_res

    def predict_first_layer(self):
        fold_list = self.fold_list
        to_preds = self.test.values
        for id,((idx_tr, idx_va),idx_test) in enumerate(fold_list):
            layer_id = f"{self.surfix}layer_1/{id}"
            self.first_models_res_return = {}
            for model in self.config["layer_1_model_list"]:
                self.first_models_res_return[model] = self.pred_single_model(model,to_preds,layer_id)

            second_layer_input_fold = self.test.values
            for idi,(model_name,oof_res) in enumerate(self.first_models_res_return.items()):

                second_layer_input_fold = np.concatenate([second_layer_input_fold,oof_res],axis=1)

            if id == 0:
                self.second_layer_input = second_layer_input_fold
            else :
                self.second_layer_input += second_layer_input_fold

        self.second_layer_input = self.second_layer_input/(len(fold_list))

        gc.collect()
        torch.cuda.empty_cache()
        print("Predicting of first layer finished")

    def predict_second_layer(self):
        fold_list = self.fold_list
        to_preds = self.second_layer_input

        for id,((idx_tr, idx_va),idx_test) in enumerate(fold_list):
            layer_id = f"{self.surfix}layer_2/{id}"
            self.second_models_oof_return = {}
            for model in self.config["layer_2_model_list"]:
                self.second_models_oof_return[model] = self.pred_single_model(model,to_preds,layer_id)
        
            for idi,(model_name,oof_res) in enumerate(self.second_models_oof_return.items()):
                if idi == 0:
                    self.third_layer_input_fold = oof_res
                else:
                    self.third_layer_input_fold = np.concatenate([self.third_layer_input_fold,oof_res],axis=1)
            
            if id == 0:
                self.third_layer_input = self.third_layer_input_fold
            else :
                self.third_layer_input += self.third_layer_input_fold

        self.third_layer_input = self.third_layer_input/(len(fold_list))

        gc.collect()
        torch.cuda.empty_cache()
        print("Predicting of second layer finished")
    
    def predict_third_layer(self):
        fold_list = self.fold_list
        to_preds = self.third_layer_input

        for id,((idx_tr, idx_va),idx_test) in enumerate(fold_list):
            layer_id = f"{self.surfix}layer_3/{id}"
            
            last_model = self.config["layer_3_model"]
            if last_model == "mlp":
                self.final_models = FinalWrapper(params=self.params_dic['final_params'])
            elif last_model == "torch":
                self.final_models = NNWrapper(params=self.params_dic['final_params'])
            elif last_model == "catboost":
                self.final_models = CatboostWrapper(params=self.params_dic['cb_params']) 
            oof_res = self.final_models.predict(self.third_layer_input_fold,layer_id)
        
            if id == 0:
                self.third_predict = oof_res
            else :
                self.third_predict += oof_res

        self.third_predict = self.third_predict/(len(fold_list))
        print("Predicting finished")

    def predict(self,dir_n = 0):
        self.surfix = f"./results/{dir_n}/"
        self.predict_first_layer()
        self.predict_second_layer()
        self.predict_third_layer()
        return self.third_predict
    
    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)