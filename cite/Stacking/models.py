import torch
import numpy as np
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor,Pool
import lightgbm as lgb
import joblib
import os

# NN
class cell(torch.nn.Module):
    def __init__(self,input_dim,out_dim,dropout=0.1):
        super().__init__()
        self.weight_1 = torch.nn.Sequential(
            
            torch.nn.Linear(input_dim,input_dim),
            # torch.nn.Mish(),
            torch.nn.Softmax(dim= -1),
        )
        self.linear_0 = torch.nn.Sequential(
            torch.nn.Linear(input_dim,input_dim),
            torch.nn.Dropout(dropout),
        ) 
        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim,input_dim),
            torch.nn.Mish(),
            )

        self.bn_1 = torch.nn.LayerNorm((input_dim))
        self.bn_2 = torch.nn.LayerNorm((out_dim))
        self.bn_3 = torch.nn.LayerNorm((out_dim))

        self.linear_2 = torch.nn.Sequential(

            torch.nn.Linear(input_dim,out_dim),
            torch.nn.Dropout(dropout),
            # torch.nn.Mish(),
            torch.nn.Linear(out_dim,out_dim),
            # torch.nn.Dropout(dropout),
            torch.nn.Mish(),
            
        )
        self.linear_3 = torch.nn.Sequential(

            torch.nn.Linear(input_dim,out_dim),
            torch.nn.Dropout(dropout),
            # torch.nn.Mish(),
            torch.nn.Linear(out_dim,out_dim),
            # torch.nn.Dropout(dropout),
            torch.nn.Mish(),
            
        )

    def forward(self,x):
        x_1 = self.linear_1(self.linear_0(x) * self.weight_1(x))
        x = self.bn_1(x_1+x)
        x = self.bn_2(x+self.linear_2(x))
        x = self.bn_3(x+self.linear_3(x))
        return x

class modules(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        input_num = config["input_num"]
        dropout = config["dropout"]
        process_dim = config["process_dim"]
        self.model = torch.nn.ModuleList()

        in_dim_array = [
            [process_dim,process_dim,process_dim,process_dim,process_dim],
            [process_dim,process_dim,process_dim,process_dim,process_dim],
            [process_dim,process_dim,process_dim,process_dim,process_dim],
            # [input_num,input_num,input_num,input_num],
        ]
        middle_dim_array = [
            [256,256,256,256],
            [256,256,256,256],
            [256,256,256,256],
            [256,256,256,256],
        ]
        self.out_dim_array = [
            [process_dim,process_dim,process_dim,process_dim,process_dim],
            [process_dim,process_dim,process_dim,process_dim,process_dim],
            [process_dim,process_dim,process_dim,process_dim,process_dim],
        ]

        for i in range(len(in_dim_array)): # 行
            temp_model = torch.nn.ModuleList()
            for j in range(len(in_dim_array[0])): # 列
                dim_in = in_dim_array[i][j]
                dim_out = self.out_dim_array[i][j]
                temp_model.append(
                    torch.nn.Sequential(
                        cell(dim_in,dim_out,dropout),
                        # cell(dim_in,dim_out,dropout),
                    )
                )
            self.model.append(temp_model)

class NN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        process_dim = config["process_dim"]
        self.projection = torch.nn.Linear(self.input_num ,240)

        self.backbone = torch.nn.Linear(self.input_num ,process_dim)
        self.embedding_1 = torch.nn.Embedding(2,process_dim)
        self.embedding_2 = torch.nn.Embedding(7,process_dim)

        self.model = modules(config)

        tail_input_dim = np.sum(np.array(self.model.out_dim_array)[-2:,-2:])
        self.tail = torch.nn.Sequential(

            torch.nn.Linear(tail_input_dim,process_dim *4),
            torch.nn.Mish(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(process_dim *4,process_dim *2),
            torch.nn.Mish(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(process_dim *2,output_num),
            torch.nn.Mish(),
        )
        
    def forward(self,xin):
        linears_1_index = 0  # 列
        linears_2_index = 0  # 行
        # xin_1 = self.projection(xin[:,:self.input_num])
        xin = self.backbone(xin[:,:self.input_num]) # + self.embedding_2(xin[:,-1].int())  # + self.embedding_1(xin[:,-2].int())# 
        # neck
        temp_model_list = self.model.model[0]
        res_array = []
        res_list  = []
        temp_out = xin
        linears_1_index += 1
        for id,i in enumerate(temp_model_list):
            if id ==0 :
                temp_out = i(
                    temp_out+xin
                    )
                linears_2_index += 1
            else:
                temp_out = i(
                    temp_out+xin
                    )
                linears_2_index += 1
            res_list.append(temp_out) 
        # res_array += res_list

        temp_model_list = self.model.model[1]
        temp_out = xin
        linears_1_index += 1
        for i in range(len(temp_model_list)):
            if i == -1:
                temp_out = temp_model_list[i](
                  res_list[i]
                )
            else:
                temp_out = temp_model_list[i](
                res_list[i]+temp_out
                    )
            res_list[i] = temp_out
        res_array += res_list[-2:]

        temp_model_list = self.model.model[2]
        temp_out = xin
        linears_1_index += 1
        for i in range(len(temp_model_list)):
            if i == -1:
                temp_out = temp_model_list[i](
                   res_list[i]
                )
            else:
                temp_out = temp_model_list[i](
                    res_list[i]+temp_out
                )
            res_list[i] = temp_out
        res_array += res_list[-2:]

        res_array = torch.concat(res_array,dim = -1)
        res_array = self.tail(res_array)

        
        # tail
        return res_array

# CNN
class conv_block(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        mlp_dims = config["mlp_dims"]

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )
        self.conv_2_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )
        self.conv_2_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 15,
                stride  = 1,
                padding = 7,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
        )

    def forward(self,x):
        x1 = self.conv_2(x)
        x2 = self.conv_2_1(x)
        x3 = self.conv_2_2(x)
        x = x1+x2+x3+x
        return x 

class CNN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        self.layers = config["layers"]

        self.backbone = torch.nn.Linear(self.input_num ,self.input_num)
        self.embedding_1 = torch.nn.Embedding(2,256)
        self.embedding_2 = torch.nn.Embedding(7,256)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.input_num,4096),
            # torch.nn.Linear(2048,4096)
        )
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            # torch.nn.Mish(),
            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )

        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            # torch.nn.Mish(),

            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )

        self.conv_1_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 15,
                stride  = 1,
                padding = 7,              
            ),
            # torch.nn.Mish(),
            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
        )

        self.conv_layers = torch.nn.ModuleList()
        for i in range(self.layers):
            self.conv_layers.append(conv_block(config))

        self.final = torch.nn.Sequential(

            torch.nn.Flatten(),
            torch.nn.Linear(4096,2048),
            torch.nn.Mish(),
            torch.nn.Linear(2048,512),
            torch.nn.Mish(),
            torch.nn.Linear(512,output_num),
            torch.nn.Mish(),
            
        )
    
    def forward(self,x):
        # x_ = self.embedding_2(x[:,-1].int())
        # x_ = torch.repeat_interleave(torch.unsqueeze(x_,-1),16,-1)
        x = self.proj(x[:,:self.input_num])
        x = torch.reshape(x,(x.shape[0],256,16))
        # x = x+x_
        x1 = self.conv_1(x)
        x2 = self.conv_1_1(x)
        x3 = self.conv_1_2(x)
        # res_list = []
        x = x1+x2+x3
        # res_list.append(x)

        for layer in self.conv_layers:
            x = layer(x)
            # res_list.append(x)

        # x = torch.concat(res_list,dim =-1)
        x = self.final(x)
        return x

# Final MLP
class final_linear(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        input_num = config["input_num"]
        dropout = config["dropout"]
        model_num = config["model_num"]
        self.model_num = model_num

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_num,output_num*3),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(output_num*3,output_num),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(output_num,model_num*5),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(model_num*5,model_num),
            torch.nn.Softmax(dim = 1)
        )
        # self.para = torch.nn.Parameter(torch.randn(model_num))
    
    def forward(self,x):
        weight= self.mlp(x)
        for i in range(self.model_num):
            tmp = x[:,i*140:(i+1)*140]* torch.repeat_interleave(torch.unsqueeze(weight[:,i],-1),140,-1)
            if i == 0:
                final = tmp
            else:
                final += tmp
        return final

# Catboost
class MultiOutputCatboostRegressor:
    def __init__(self,params):
        self.params = params
        self.model_list = []

    def fit(self,train_data,train_label,val_data,val_label,**fit_params):
        
        output_num = train_label.shape[1]
        for i in tqdm(range(output_num),leave=False):
            model = CatBoostRegressor(**self.params)
            train_pool = Pool(train_data,train_label[:,i])
            model.fit(train_pool,
                eval_set = (val_data,val_label[:,i]),
                early_stopping_rounds = 10,
                verbose_eval = 0,
                use_best_model = True,
                )
            self.model_list.append(model)
            
    def predict(self,test_data):
        test_data = Pool(test_data)
        res_list = []
        for model in tqdm(self.model_list,leave=False):
            res = model.predict(test_data,thread_count = 19)
            res_list.append(res)
        res_list = np.stack(res_list,axis = 1)
        return res_list
        
    def dump(self,path = "./models/MOCB/" ):
        count = 0
        os.makedirs(path,exist_ok=True)
        for model in tqdm(self.model_list,leave=False):
            model_path = f'{path}model_{str(count)}.cbm'
            model.save_model(model_path)
            count += 1
        print("Model saved")

    def load(self,path = "./models/MOCB/" ):
        models = os.listdir(path)
        if len(self.model_list) != 0:
            raise ValueError("Don't load!")
        else:
            for i in tqdm(range(len(models)),leave=False):
                # print("begin")
                model = CatBoostRegressor(self.params)
                model_path = f'{path}model_{i}.cbm'
                model.load_model(model_path)
                # print("end")
                self.model_list.append(model)
            print("Model loaded")

# LGBM
class MultiOutputLGBMRegressor:
    def __init__(self,params):
        self.params = params
        self.model_list = []

    def fit(self,train_data,train_label,val_data,val_label,**fit_params):
        output_num = train_label.shape[1]
        for i in tqdm(range(output_num),leave=False):
            train_set = lgb.Dataset(train_data,train_label[:,i])
            val_set = lgb.Dataset(val_data,val_label[:,i])
            model = lgb.train(
                self.params,
                train_set,
                valid_sets = val_set,
                callbacks=[
                    lgb.early_stopping(20,verbose = False),
                    # lgb.log_evaluation(100),
                ]
            )
            self.model_list.append(model)
            
    def predict(self,test_data):
        res_list = []
        for model in tqdm(self.model_list,leave=False):
            res = model.predict(test_data)
            res_list.append(res)
        res_list = np.stack(res_list,axis = 1)
        return res_list
        
    def dump(self,path = "./models/MOLGB/" ):
        count = 0
        os.makedirs(path,exist_ok=True)
        for model in tqdm(self.model_list,leave=False):
            joblib.dump(model, f'{path}model_{str(count)}.pkl')
            count += 1
        print("Model saved")

    def load(self,path = "./models/MOLGB/" ):
        models = os.listdir(path)
        if len(self.model_list) != 0:
            raise ValueError("Don't load!")
        else:
            for i in tqdm(range(len(models)),leave=False):
                # print("begin")
                model = joblib.load(f'{path}model_{i}.pkl')
                # print("end")
                self.model_list.append(model)
            print("Model loaded")


