from __future__ import print_function

import os

import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler,HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

import time
from tqdm import tqdm
import gc
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GroupKFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
import os
os.environ["RAY_record_ref_creation_sites"] = "1"




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

class TrainMNIST(tune.Trainable):
    def setup(self, config):
        # data = config.pop("data")
        self.prepare_data()
        self.config = config      

    def step(self):
        np.random.seed(42)
        random.seed(42)
        kf = GroupKFold(n_splits=3) 
        index = 0
        mean_score = []
        for id,(idx_tr, idx_va) in enumerate(kf.split(range(self.train_.shape[0]),groups= self.meta_train.donor)):
            
            Xtr, Xva = self.train_[idx_tr], self.train_[idx_va]
            Ytr, Yva = self.target[idx_tr], self.target[idx_va]
            gc.collect()
            random.seed(42)
            index_to_train = random.choices([i for i in range(Xtr.shape[0])],k = 5000)
            Xtr = Xtr[index_to_train]
            Ytr = Ytr[index_to_train]
            # print(f'Fold {id}..')
        
            kernel = RBF(self.config["gamma"])
            model = KernelRidge(kernel=kernel,alpha=self.config["alpha"])
            model.fit(Xtr,Ytr)
        
            del Xtr, Ytr
            gc.collect()
            s = correlation_score(Yva, model.predict(Xva))
            mean_score.append(s)
            del Xva, Yva
            gc.collect()

            index += 1
        mean_score = np.mean(mean_score)

        return {"mean_score": mean_score}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        pass
    
    def prepare_data(self):
        self.meta_train = pd.read_pickle("~/meta_train.pkl")
        self.train_ = pd.read_pickle("~/train_.pkl").values
        self.target = pd.read_pickle("~/target.pkl").values

def train():
    sched = HyperBandForBOHB(metric="mean_score", mode="max")
    algo = TuneBOHB(metric="mean_score", mode="max")

    tuner = tune.Tuner(
        tune.with_resources(TrainMNIST, resources={"cpu": 2, "gpu":0 }),
        run_config=air.RunConfig(
            stop={
                "mean_score": 0.90,
            },
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False, checkpoint_frequency=0
            ),
            failure_config = air.FailureConfig(
                max_failures = 0,
            )
        ),

        tune_config=tune.TuneConfig(
            scheduler=sched,
            num_samples=30,
            search_alg = algo
        ),
        
        param_space=dict(

            alpha = tune.quniform(0, 3, 0.1),
            gamma = tune.randint(0, 40)
            
        ),
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)


if __name__ == "__main__":
    try:
        ray.init(address="ray://192.168.31.84:10001")
        train()
        ray.shutdown()
    except:
        print("stop")
        ray.shutdown()

'''
(TunerInternal pid=46700) +---------------------+------------+----------------------+---------+---------+--------+------------------+--------------+
(TunerInternal pid=46700) | Trial name          | status     | loc                  |   alpha |   gamma |   iter |   total time (s) |   mean_score |
(TunerInternal pid=46700) |---------------------+------------+----------------------+---------+---------+--------+------------------+--------------|
(TunerInternal pid=46700) | TrainMNIST_b4548adf | RUNNING    | 192.168.31.84:24908  |     0.9 |      35 |     10 |         2064.35  |     0.887322 |
(TunerInternal pid=46700) | TrainMNIST_b45d9ff2 | RUNNING    | 192.168.31.120:1964  |     0.8 |      37 |     11 |         2924.7   |     0.887343 |
(TunerInternal pid=46700) | TrainMNIST_bd2bd021 | RUNNING    | 192.168.31.84:15408  |     0.8 |      35 |     10 |         2291.93  |     0.887346 |
'''