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
#%% md
### Init Session
#%%
session_num = '005'
#%%
outputs_folder_path_csv = 'output_files_MNIST/csv'
outputs_folder_path_txt = 'output_files_MNIST/txt'
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
def objective(trial: Trial, logger: Logger):
    # Define Hyperparameters - Structure HPs
    activation = trial.suggest_categorical('activation_fn', ['relu', 'sigmoid', 'tanh'])
    num_hidden_layer = trial.suggest_int('num_hidden_layer', 3, 3)

    network_architecture = [28 * 28]
    for i in range(num_hidden_layer):
        layer_width = trial.suggest_int(f'hidden_layer_n{i+1}_size', 0, 128, 8)
        network_architecture.append(layer_width)
    network_architecture.append(10)

    # Define Hyperparameters - Training HPs
    batch_size = trial.suggest_int('batch_size', 16, 64, 16)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    loss_function_str = trial.suggest_categorical('loss_fn', ['CrossEntropy', 'Focal'])
    optimizer_str = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])

    # Define Hyperparameters - Epochs
    max_epochs = trial.suggest_int('max_epochs', 20, 20)


    # Init DataLoaders
    train_loader = init_data_loader(train_dataset, batch_size=batch_size)
    val_loader = init_data_loader(val_dataset, batch_size=batch_size)
    test_loader = init_data_loader(test_dataset, batch_size=batch_size)

    # Init Model
    model_extra_args = {'network_architecture': network_architecture, 'activation': activation}
    model = init_model(model_str='MLP', extra_args=model_extra_args).to(get_device())

    # Init Loss
    loss_fn = get_loss_fn(loss_str=loss_function_str)

    # Init Optimizer
    optimizer = get_optimizer(model=model, optimizer_str=optimizer_str, learning_rate=learning_rate)

    # Init Regularizer
    regularizer = Regularizer(lambda_depth=0.1, lambda_tot_widths=0.4, max_depth=3, max_width=128)

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
                             n_trials=128,
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
HyperbandPruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=20, reduction_factor=3, bootstrap_count=4)
#%% md
### Run Optimizations
#%% md
#### Random Sampler
#%%
study_Random = optuna.create_study(direction=DIRECTION, sampler=RandomSampler, pruner=HyperbandPruner)
optuna_runner(study_Random, 'Random_Sampler')
#%% md
#### TPE Sampler
#%%
study_TPE = optuna.create_study(direction=DIRECTION, sampler=TPESampler, pruner=HyperbandPruner)
optuna_runner(study_TPE, 'TPE_Sampler')
#%% md
## Display Best Results
#%%
# Alias for Best Study
best_study = study_Random
#%%
from utils.persistency.file_name_builder import file_name_builder, folder_exists_check

# Init DataLoaders
train_loader_best = init_data_loader(train_dataset, batch_size=best_study.best_params['batch_size'])
val_loader_best = init_data_loader(val_dataset, batch_size=best_study.best_params['batch_size'])
test_loader_best = init_data_loader(test_dataset, batch_size=best_study.best_params['batch_size'])

# Init Model
best_hidden_layer_sizes = [28*28] + [best_study.best_params[f'hidden_layer_n{i+1}_size'] for i in range(best_study.best_params['num_hidden_layer'])] + [10]
best_model_extra_args = {'network_architecture': best_hidden_layer_sizes,
                         'activation': get_activation_fn(best_study.best_params['activation_fn'])}
best_model = init_model(model_str='MLP', extra_args=best_model_extra_args).to(get_device())

# Init Loss and Optimizer
loss_fn_best = get_loss_fn(loss_str=best_study.best_params['loss_fn'])
optimizer_best = get_optimizer(model=best_model,
                               optimizer_str=best_study.best_params['optimizer'],
                               learning_rate=best_study.best_params['learning_rate'])

# Init Early Stopper
early_stopper_best = EarlyStopper(patience=5, mode="maximize")

# Init Logger
folder_exists_check(outputs_folder_path_txt, session_num, f'log_BEST_STUDY')
logger_best_study = Logger(file_name_builder(outputs_folder_path_txt, session_num, f'log_BEST_STUDY', 'txt'))


# Train Model
full_train_loop(max_epochs=best_study.best_params['max_epochs'],
                train_loader=train_loader_best, val_loader=val_loader_best, test_loader=test_loader_best,
                model=best_model,
                loss_fn=loss_fn_best,
                optimizer=optimizer_best,
                regularizer=None,
                early_stopper=early_stopper_best,
                logger=logger_best_study,
                trial=None)
#%%
images, y_pred_best, y_test = prediction_loop(test_loader_best, best_model)
#%%
display_images(images, y_pred_best, y_test, 50)