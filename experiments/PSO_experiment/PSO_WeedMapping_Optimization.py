#%% md
# Particle Swarm Optimization - Weed Mapping
#%% md
## Environment Setup
#%% md
### Import Dependencies
#%%
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
# sys.path.insert(0, '../code/Users/f.chinnicarella/src/root_workspace/Bachelor-Thesis')

from utils.persistency.logger import Logger

from utils.dataset.build_dataset import load_weedmap_data
from utils.dataset.build_dataloader import init_data_loaders_weedmapping

from utils.training.train_loop import full_train_loop_weedmapping
from utils.model.model_utils import init_model
from utils.optimization.early_stopper import EarlyStopper
from utils.optimization.regularizer import MODEL_ARCHITECTURES_WEEDMAPPING
from utils.optimization.regularizer import Regularizer_WeedMapping
from utils.misc.device import get_device
from utils.model.model_utils import get_loss_fn, get_optimizer
from experiments.PSO_experiment.backend.PSO import PSO, PSOTrial
from experiments.PSO_experiment.backend.pso_utils import decode_hyperparameter, build_encoded_dict
from experiments.PSO_experiment.backend.pso_utils import BACKBONE_BOUNDS, OPTIMIZER_BOUNDS
from experiments.PSO_experiment.backend.pso_runner import PSORunner
from experiments.PSO_experiment.backend.pso_pruners import PSOMedianPruner
#%% md
### Init Session
#%%
session_num = '000'
#%%
outputs_folder_path_csv = 'output_files_PSO_WeedMapping/csv'
outputs_folder_path_txt = 'output_files_PSO_WeedMapping/txt'
#%% md
## Load Data
#%%
weed_mapping_dataset = load_weedmap_data()
#%% md
## Optuna Optimization
#%% md
### Define Objective Function
#%%
def objective(trial: PSOTrial, logger: Logger):
    # Define Hyperparameters - Structure HPs
    backbone_str = decode_hyperparameter(build_encoded_dict(trial, BACKBONE_BOUNDS))
    # backbone_str = 'MiT-B0'

    network_architecture = MODEL_ARCHITECTURES_WEEDMAPPING[backbone_str]
    trial.set_user_attr('network', network_architecture)


    # Define Hyperparameters - Training HPs - Batch Sizes
    # batch_size_train = round(trial.hyperparameters['batch_size_train'])
    # batch_size_val = round(trial.hyperparameters['batch_size_val'])
    batch_size_train = 4
    batch_size_val = 4

    # Define Hyperparameters - Training HPs - Learning Rate
    learning_rate = trial.hyperparameters['learning_rate']
    # learning_rate = 1e-3

    # Define Hyperparameters - Training HPs - Optimizer
    optimizer_str = decode_hyperparameter(build_encoded_dict(trial, OPTIMIZER_BOUNDS))
    # optimizer_str = 'Adam'
    trial.set_user_attr('optimizer', optimizer_str)

    # Define Hyperparameters - Training HPs - Loss Function
    # loss_gamma = round(trial.hyperparameters['loss_gamma'])
    # loss_weight = [trial.hyperparameters[f'loss_weight_{i+1}'] for i in range(3)]
    loss_gamma = 2.0
    loss_weight = [0.06, 1.0, 1.7]

    # Define Hyperparameters - Max Epochs
    max_epochs = 30


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
    early_stopper = EarlyStopper(patience=5, mode="maximize")


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
#### PSO Constants
#%%
ATTRS = ('generation', 'particle_id', 'hp_num_hidden_layer', 'score', 'user_attrs_epochs', 'user_attrs_network', 'user_attr_accuracy', 'user_attr_precision', 'user_attr_recall', 'user_attr_f1', 'state', 'duration', 'hp_hidden_layer_n1_size', 'hp_hidden_layer_n2_size', 'hp_hidden_layer_n3_size', 'hp_learning_rate' 'datetime_start', 'datetime_complete')
#%%
DIRECTION = 'maximize'
#%% md
### Define Study
#%%
DYNAMIC_HPs = {
    **BACKBONE_BOUNDS,

    # 'batch_size_train': [4, 8],
    # 'batch_size_val': [6, 12],

    'learning_rate': [1e-4, 1e-2],
    **OPTIMIZER_BOUNDS,

    # 'loss_gamma': [0.5, 5.0],
    # 'loss_weight_1': [0.1, 2.0],
    # 'loss_weight_2': [0.1, 2.0],
    # 'loss_weight_3': [0.1, 2.0],
}
#%%
pso_pruner = PSOMedianPruner(n_startup_generations=3, n_warmup_steps=4, interval_steps=4, min_trials_per_step=4)
#%%
pso = PSO(objective_fn=objective, hps_bounds=DYNAMIC_HPs, num_particles=8, max_generations=10, pruner=None)
#%% md
### Run Optimization
#%%
pso_runner = PSORunner(path_csv=outputs_folder_path_csv,
                       path_txt=outputs_folder_path_txt,
                       session_num=session_num,
                       n_jobs=2,
                       metric_to_follow='f1', attrs=None)
#%%
pso_runner(pso, 'PSO_Optimization_WeedMapping')