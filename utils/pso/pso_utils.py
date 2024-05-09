from utils.pso.PSO import PSOTrial
from utils.optimization.regularizer import MODEL_ARCHITECTURES_WEEDMAPPING


ACTIVATION_FN_BOUNDS = {
    'sigmoid': [0, 1],
    'relu': [0, 1],
    'tanh': [0, 1],
}

LOSS_FN_BOUNDS = {
    'CrossEntropy': [0, 1],
    'Focal': [0, 1],
}

OPTIMIZER_BOUNDS = {
    'SGD': [0, 1],
    'Adam': [0, 1],
}

BACKBONE_BOUNDS = {
    key: [0, 1] for key in MODEL_ARCHITECTURES_WEEDMAPPING.keys()
}


def build_encoded_dict(trial: PSOTrial, hyperparameters_bounds: dict):
    return {key: trial.hyperparameters[key] for key in hyperparameters_bounds.keys()}


def decode_hyperparameter(hyperparameter_encodings: dict):
    max_key = max(hyperparameter_encodings, key=hyperparameter_encodings.get)
    return max_key
