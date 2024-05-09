#%% md
# MNIST Optuna Optimization
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

from utils.dataset.build_dataset import load_MNIST_data
from utils.dataset.build_dataloader import init_data_loader

from utils.training.train_loop import full_train_loop
from utils.model.model_utils import init_model
from utils.optimization.early_stopper import EarlyStopper
from utils.optimization.regularizer import Regularizer
from utils.misc.device import get_device
from utils.model.model_utils import get_activation_fn, get_loss_fn, get_optimizer
from utils.optimization.optuna_runner import OptunaRunner
from utils.display_results.display_results import prediction_loop
from utils.display_results.display_results import display_images
from utils.persistency.file_name_builder import file_name_builder, folder_exists_check
#%% md
### Init Session
#%%
session_num = '010'
#%%
outputs_folder_path_csv = 'output_files_MNIST/csv'
outputs_folder_path_txt = 'output_files_MNIST/txt'
#%% md
## Load Data
#%%
train_dataset, val_dataset, test_dataset = load_MNIST_data('data_MNIST/')
#%% md
## Optuna Optimization
#%% md
### Define Objective Function
#%%
def objective(trial: Trial, logger: Logger):
    # Define Hyperparameters - Structure HPs - Activation Function
    activation = trial.suggest_categorical('activation_fn', ['relu', 'sigmoid', 'tanh'])
    # activation = 'relu'

    # Define Hyperparameters - Structure HPs - Network Architecture (Depth)
    # num_hidden_layer = trial.suggest_int('num_hidden_layer', 3, 3)
    num_hidden_layer = 3

    # Define Hyperparameters - Structure HPs - Network Architecture (Width)
    network_architecture = [28 * 28]
    for i in range(num_hidden_layer):
        layer_width = trial.suggest_int(f'hidden_layer_n{i+1}_size', 0, 128, 8)
        if layer_width != 0:
            network_architecture.append(layer_width)
    network_architecture.append(10)
    trial.set_user_attr('network', network_architecture)


    # Define Hyperparameters - Training HPs - Batch Size
    # batch_size = trial.suggest_int('batch_size', 16, 64, 16)
    batch_size = 16

    # Define Hyperparameters - Training HPs - Learning Rate
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    # learning_rate = 1e-3

    # Define Hyperparameters - Training HPs - Loss Function
    # loss_function_str = trial.suggest_categorical('loss_fn', ['CrossEntropy', 'Focal'])
    loss_function_str = 'CrossEntropy'

    # Define Hyperparameters - Training HPs - Optimizer
    optimizer_str = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    # optimizer_str = 'Adam'

    # Define Hyperparameters - Max Epochs
    max_epochs = 30


    # Init DataLoaders
    train_loader = init_data_loader(train_dataset, batch_size=batch_size)
    val_loader = init_data_loader(val_dataset, batch_size=batch_size)
    test_loader = init_data_loader(test_dataset, batch_size=batch_size)

    # Init Model
    model_extra_args = {'network_architecture': network_architecture, 'activation': get_activation_fn(activation)}
    model = init_model(model_str='MLP', extra_args=model_extra_args).to(get_device())

    # Init Loss
    loss_fn = get_loss_fn(loss_str=loss_function_str)

    # Init Optimizer
    optimizer = get_optimizer(model=model, optimizer_str=optimizer_str, learning_rate=learning_rate)

    # Init Regularizer
    regularizer = Regularizer(lambda_tot_widths=0.4, max_depth=3, max_width=128)

    # Init Early Stopper
    early_stopper = EarlyStopper(patience=5, mode="maximize")


    # Perform Training
    optim_score = full_train_loop(max_epochs=max_epochs,
                                  train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                  model=model,
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
                             n_jobs=-1,
                             n_trials=256,
                             path_csv=outputs_folder_path_csv,
                             path_txt=outputs_folder_path_txt,
                             session_num=session_num,
                             metric_to_follow='accuracy',
                             attrs=ATTRS)
#%% md
#### Optuna Constants - Samplers
#%%
RandomSampler = optuna.samplers.RandomSampler()
TPESampler = optuna.samplers.TPESampler()
#%% md
#### Optuna Constants - Pruners
#%%
MedianPruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=1, interval_steps=2, n_min_trials=4)
HyperbandPruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=30, reduction_factor=3, bootstrap_count=6)
#%% md
### Run Optimizations
#%% md
#### Random Sampler
#%%
study_Random = optuna.create_study(direction=DIRECTION, sampler=RandomSampler, pruner=None)
optuna_runner(study_Random, 'Random_Sampler')
#%% md
#### TPE Sampler
#%%
study_TPE = optuna.create_study(direction=DIRECTION, sampler=TPESampler, pruner=None)
optuna_runner(study_TPE, 'TPE_Sampler')