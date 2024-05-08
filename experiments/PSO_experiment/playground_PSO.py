#%% md
# Particle Swarm Optimization
#%% md
## Environment Setup
#%% md
### Import Dependencies
#%%
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from utils.persistency.logger import Logger

from utils.dataset.build_dataset import load_MNIST_data
from utils.dataset.build_dataloader import init_data_loader

from utils.training.train_loop import full_train_loop
from utils.model.model_utils import init_model
from utils.optimization.early_stopper import EarlyStopper
from utils.optimization.regularizer import Regularizer
from utils.misc.device import get_device
from utils.model.model_utils import get_activation_fn, get_loss_fn, get_optimizer
from utils.pso.PSO import PSO, PSOTrial
from utils.pso.pso_runner import PSORunner
from utils.pso.pso_pruners import PSOMedianPruner

from utils.display_results.display_results import prediction_loop
from utils.display_results.display_results import display_images
from utils.persistency.file_name_builder import file_name_builder, folder_exists_check
#%% md
### Init Session
#%%
session_num = '003'
#%%
outputs_folder_path_csv = 'output_files_PSO/csv'
outputs_folder_path_txt = 'output_files_PSO/txt'
#%% md
## Load Data
#%%
train_dataset, val_dataset, test_dataset = load_MNIST_data('data_MNIST/')
#%%
# train_loader = init_data_loader(train_dataset, batch_size=32)
# val_loader = init_data_loader(val_dateset, batch_size=32)
# test_loader = init_data_loader(test_dataset, batch_size=32)
#%% md
## Optuna Optimization
#%% md
### Define Objective Function
#%%
def objective(trial: PSOTrial, logger: Logger):
    # Define Hyperparameters - Structure HPs
    activation = 'relu'
    # num_hidden_layer = round(trial.hyperparameters['num_hidden_layer'])
    num_hidden_layer = 3

    network_architecture = [28 * 28]
    for i in range(num_hidden_layer):
        layer_width = round(trial.hyperparameters[f'hidden_layer_n{i+1}_size'])
        if layer_width > 8:
            network_architecture.append(layer_width)
    network_architecture.append(10)
    trial.set_user_attr('network', network_architecture)

    # Define Hyperparameters - Training HPs
    batch_size = 64
    learning_rate = trial.hyperparameters['learning_rate']
    loss_function_str = 'CrossEntropy'
    optimizer_str = 'Adam'

    # Define Hyperparameters - Epochs
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
#### PSO Constants
#%%
ATTRS = ('generation', 'particle_id', 'hp_num_hidden_layer', 'score', 'user_attrs_epochs', 'user_attrs_network', 'user_attr_accuracy', 'user_attr_precision', 'user_attr_recall', 'user_attr_f1', 'state', 'duration', 'hp_hidden_layer_n1_size', 'hp_hidden_layer_n2_size', 'hp_hidden_layer_n3_size', 'hp_learning_rate' 'datetime_start', 'datetime_complete')
#%%
DIRECTION = 'maximize'
#%% md
### Define Study
#%%
DYNAMIC_HPs = {
    # 'num_hidden_layer': [3, 3],
    'hidden_layer_n1_size': [0, 128],
    'hidden_layer_n2_size': [0, 128],
    'hidden_layer_n3_size': [0, 128],
    'learning_rate': [1e-4, 1e-2]
}
#%%
pso_pruner = PSOMedianPruner(n_startup_generations=3, n_warmup_steps=4, interval_steps=4, min_trials_per_step=4)
#%%
pso = PSO(objective_fn=objective, hps_bounds=DYNAMIC_HPs, num_particles=20, max_generations=10, pruner=None)
#%% md
### Run Optimization
#%%
pso_runner = PSORunner(path_csv=outputs_folder_path_csv,
                       path_txt=outputs_folder_path_txt,
                       session_num=session_num,
                       n_jobs=10,
                       metric_to_follow='accuracy', attrs=None)
#%%
pso_runner(pso, 'PSO_Optimization')