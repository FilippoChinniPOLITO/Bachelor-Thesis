#%% md
# Weed Mapping Optuna Optimization
#%% md
## Environment Setup
#%% md
### Import Dependencies
#%%
import optuna
from optuna import Trial

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('optuna').setLevel(logging.WARNING)

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
# sys.path.insert(0, '../code/Users/f.chinnicarella/src/root_workspace/Bachelor-Thesis')


from utils.persistency.logger import Logger

from utils.dataset.build_dataset import load_weedmap_data
from utils.dataset.build_dataloader import init_data_loaders_weedmapping

from utils.training.train_loop import full_train_loop, full_train_loop_weedmapping
from utils.model.model_utils import init_model
from utils.optimization.early_stopper import EarlyStopper
from utils.optimization.regularizer import Regularizer, Regularizer_WeedMapping, MODEL_ARCHITECTURES_WEEDMAPPING
from utils.misc.device import get_device
from utils.model.model_utils import get_activation_fn, get_loss_fn, get_optimizer
from utils.optimization.optuna_runner import OptunaRunner
from utils.display_results.display_results import prediction_loop
from utils.display_results.display_results import display_images
#%% md
### Init Session
#%%
session_num = '000'
#%%
outputs_folder_path_csv = 'output_files_weed_mapping/csv'
outputs_folder_path_txt = 'output_files_weed_mapping/txt'
#%% md
## Load Data
#%%
weed_mapping_dataset = load_weedmap_data()
#%%
# _tuple = init_data_loaders_weedmapping(weed_mapping_dataset, batch_size_train=6, batch_size_val=12, batch_size_test=12)
# train_loader, val_loader, test_loader = _tuple
#%% md
## Optuna Optimization
#%% md
### Define Objective Function
#%%
def objective(trial: Trial, logger: Logger):
    # Define Hyperparameters - Structure HPs
    backbone_str = trial.suggest_categorical('backbone', [s for s in MODEL_ARCHITECTURES_WEEDMAPPING.keys()])

    # Define Hyperparameters - Batch Sizes
    # batch_size_train = trial.suggest_int('batch_size_train', 4, 8, 2)
    # batch_size_val = trial.suggest_int('batch_size_val', 6, 12, 6)
    batch_size_train = 4
    batch_size_val = 4

    # Define Hyperparameters - Training HPs
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_str = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])

    # Define Hyperparameters - Loss Function
    # loss_gamma = trial.suggest_float('loss_gamma', 0.5, 5.0, log=True)
    # loss_weight = [trial.suggest_float(f'loss_weight_{i+1}', 0.1, 2.0, log=True) for i in range(3)]
    loss_gamma = 2.0
    loss_weight = [0.06, 1.0, 1.7]

    # Define Hyperparameters - Epochs
    max_epochs = 200


    # Init DataLoaders
    train_loader, val_loader, test_loader = init_data_loaders_weedmapping(dataset=weed_mapping_dataset,
                                                                          batch_size_train=batch_size_train,
                                                                          batch_size_val=batch_size_val,
                                                                          batch_size_test=batch_size_val)

    # Init Model
    model_extra_args = {"backbone": backbone_str, "input_channels": 5}
    model = init_model(model_str='Lawin', extra_args=model_extra_args).to(get_device())

    # Init Loss
    focal_extra_args = {"gamma": loss_gamma, "weight": loss_weight}
    loss_fn = get_loss_fn(loss_str='Focal', extra_args=focal_extra_args)

    # Init Optimizer
    optimizer = get_optimizer(model=model, optimizer_str=optimizer_str, learning_rate=learning_rate)

    # Init Regularizer
    regularizer = Regularizer_WeedMapping(lambda_widths=0.4, max_sum_widths=1024)

    # Init Early Stopper
    early_stopper = EarlyStopper(patience=15, mode="maximize")


    # Perform Training
    optim_score = full_train_loop_weedmapping(max_epochs=max_epochs,
                                              train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                              model=model,
                                              backbone_str=backbone_str,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              regularizer=regularizer,
                                              early_stopper=early_stopper,
                                              logger=logger,
                                              trial=trial)

    return optim_score
#%% md
#### Optuna Constants - Study Parameters
#%%
ATTRS = ('number', 'value', 'user_attrs', 'state', 'params', 'duration', 'datetime_start', 'datetime_complete')
#%%
DIRECTION = 'maximize'
#%%
optuna_runner = OptunaRunner(objective_fn=objective,
                             n_jobs=2,
                             n_trials=64,
                             path_csv=outputs_folder_path_csv,
                             path_txt=outputs_folder_path_txt,
                             session_num=session_num,
                             metric_to_follow='f1',
                             attrs=ATTRS)
#%% md
#### Optuna Constants - Samplers
#%%
RandomSampler = optuna.samplers.RandomSampler()
TPESampler = optuna.samplers.TPESampler()
#%% md
#### Optuna Constants - Pruners
#%%
MedianPruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=5, interval_steps=25, n_min_trials=4)
HyperbandPruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=200, reduction_factor=3, bootstrap_count=4)
#%% md
### Run Optimizations
#%% md
#### Random Sampler
#%%
study_Random = optuna.create_study(direction=DIRECTION, sampler=RandomSampler, pruner=MedianPruner)
optuna_runner(study_Random, 'Random_Sampler')
#%% md
#### TPE Sampler
#%%
study_TPE = optuna.create_study(direction=DIRECTION, sampler=TPESampler, pruner=HyperbandPruner)
optuna_runner(study_TPE, 'TPE_Sampler')