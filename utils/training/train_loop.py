import optuna
from torch import cuda
from optuna import Trial

from experiments.weed_mapping_experiment.backend.model.lawin_model import Lawin
from utils.training.train_step import train_step
from utils.training.eval_step import eval_step


def full_train_loop(max_epochs, train_loader, val_loader, test_loader, model, loss_fn, optimizer, early_stopper, regularizer, logger, trial: Trial = None):
    # Init Special Parameters
    is_optuna = False
    pso_attributes = (False, None, None)
    if trial is not None:
        is_optuna = True
        pso_attributes = get_pso_attributes(trial)
    if regularizer is None:
        regularizer = lambda **kwargs: kwargs.get('score', None)

    # Get Metric to Follow
    metric_index = get_metric_to_follow(model)


    # Train Model
    for epoch_index in range(max_epochs):

        # Training Step
        train_step(train_loader, model, loss_fn, optimizer)

        # Evaluation Step (Intermediate)
        val_loss, val_metrics = eval_step(val_loader, model, loss_fn)
        val_metrics_processed = [i.item() for i in val_metrics]

        # Intermediate Optimization Score
        optim_score = regularizer(score=val_metrics_processed[metric_index],
                                  network_architecture=model.get_network_architecture())

        # Print Intermediate Evaluation
        intermediate_reporting(val_loss=val_loss, val_metrics=val_metrics_processed,
                               intermediate_score=optim_score, epoch_index=epoch_index,
                               logger=logger, is_optuna=is_optuna, trial=trial, pso_attributes=pso_attributes)

        # Check Early-Stopping Step
        if early_stopper(score=optim_score, model=model):
            early_stopping_reporting(logger=logger, is_optuna=is_optuna, trial=trial, pso_attributes=pso_attributes)
            break

        # Pruning Step
        pruning_step(val_metrics=val_metrics_processed,
                     intermediate_score=optim_score, epoch_index=epoch_index,
                     logger=logger, is_optuna=is_optuna, trial=trial, pso_attributes=pso_attributes)

    # Complete Training
    completing_reporting(logger=logger, is_optuna=is_optuna, trial=trial, pso_attributes=pso_attributes)
    model.load_state_dict(early_stopper.get_best_model_params())


    # Evaluate Model (TestSet) - Evaluation Metrics
    test_loss, test_metrics = eval_step(test_loader, model, loss_fn)
    test_metrics_processed = [i.item() for i in test_metrics]

    # Evaluate Model - Optimization Score
    final_optim_score = regularizer(score=test_metrics_processed[metric_index],
                                    network_architecture=model.get_network_architecture())

    # Print Final Evaluation
    final_evaluation_reporting(test_metrics=test_metrics_processed,
                               final_score=final_optim_score,
                               logger=logger, is_optuna=is_optuna, trial=trial, pso_attributes=pso_attributes)

    # Empty Cache
    cuda.empty_cache()

    return final_optim_score


def get_pso_attributes(trial):
    if trial is None:
        return False, None, None

    is_pso = False
    generation = None
    particle = None
    if ('generation' in trial.system_attrs.keys()) and ('particle' in trial.system_attrs.keys()):
        is_pso = True
        generation = trial.system_attrs['generation']
        particle = trial.system_attrs['particle']

    return is_pso, generation, particle


def intermediate_reporting(val_loss, val_metrics, intermediate_score, epoch_index, logger, is_optuna=False, trial=None, pso_attributes=None):
    is_pso, generation, particle = pso_attributes
    optuna_addition_1 = ''
    optuna_addition_2 = ''

    if is_pso:
        trial.set_user_attr(key='epochs', value=epoch_index+1)
        optuna_addition_1 = f"Trial n°{trial.number} - Gen n°{generation} - Particle n°{particle} - "
        optuna_addition_2 = f" (Gen n°{generation} - Particle n°{particle})"
    elif is_optuna:
        trial.set_user_attr(key='epochs', value=epoch_index+1)
        optuna_addition_1 = f"Trial n°{trial.number} - "
        optuna_addition_2 = f" (Trial n°{trial.number})"

    logger.log(f"\n{optuna_addition_1}Epoch {epoch_index+1}\n----------------------------------------")
    logger.log(f"Intermediate Avg loss:   {val_loss:>0.4f}")
    logger.log(f"Intermediate Accuracy:   {val_metrics[0]*100:>0.4f}%")
    logger.log(f"Intermediate Precision:  {val_metrics[1]*100:>0.4f}%")
    logger.log(f"Intermediate Recall:     {val_metrics[2]*100:>0.4f}%")
    logger.log(f"Intermediate F1:         {val_metrics[3]*100:>0.4f}%\n")

    logger.log(f'Intermediate Optimization Score{optuna_addition_2}: {intermediate_score*100:>0.4f}%\n')


def early_stopping_reporting(logger, is_optuna=False, trial=None, pso_attributes=None):
    is_pso, generation, particle = pso_attributes
    if is_pso:
        logger.log(f"Trial n°{trial.number} - Gen n°{generation} - Particle n°{particle} Early Stopped!")
    elif is_optuna:
        logger.log(f"Trial n°{trial.number} Early Stopped!")
    else:
        logger.log(f"Training Early Stopped!")


def pruning_step(val_metrics, intermediate_score, epoch_index, logger, is_optuna=False, trial=None, pso_attributes=None):
    pso_addition = ''
    is_pso, generation, particle = pso_attributes

    if is_pso:
        pso_addition = f" - Gen n°{generation} - Particle n°{particle}"

    if is_optuna:
        trial.report(value=intermediate_score, step=epoch_index)
        if (intermediate_score < 0) or (trial.should_prune()):
            trial.set_user_attr(key='accuracy', value=round(val_metrics[0], 4))
            trial.set_user_attr(key='precision', value=round(val_metrics[1], 4))
            trial.set_user_attr(key='recall', value=round(val_metrics[2], 4))
            trial.set_user_attr(key='f1', value=round(val_metrics[3], 4))

            logger.log(f"Trial n°{trial.number}{pso_addition} Pruned!\n\n")

            cuda.empty_cache()
            raise optuna.TrialPruned()


def completing_reporting(logger, is_optuna=False, trial=None, pso_attributes=None):
    is_pso, generation, particle = pso_attributes
    if is_pso:
        logger.log(f"Training Gen n°{generation} - Particle n°{particle} Complete!\n\n")
    elif is_optuna:
        logger.log(f"Training n°{trial.number} Complete!\n\n")
    else:
        logger.log(f"Training Complete!\n\n")


def final_evaluation_reporting(test_metrics, final_score, logger, is_optuna=False, trial=None, pso_attributes=None):
    pso_addition = ''
    is_pso, generation, particle = pso_attributes

    if is_pso:
        pso_addition = f" - Gen n°{generation} - Particle n°{particle}"

    if is_optuna:
        trial.set_user_attr(key='accuracy', value=round(test_metrics[0], 4))
        trial.set_user_attr(key='precision', value=round(test_metrics[1], 4))
        trial.set_user_attr(key='recall', value=round(test_metrics[2], 4))
        trial.set_user_attr(key='f1', value=round(test_metrics[3], 4))
        logger.log(f"\nReport Finished Trial:\n----------------------------------------")
        logger.log(f"Trial n°{trial.number}{pso_addition}")
        logger.log(f"Hyperparameters: {trial.params}\n")

    logger.log(f"Test Accuracy:   {test_metrics[0]*100:>0.4f}%")
    logger.log(f"Test Precision:  {test_metrics[1]*100:>0.4f}%")
    logger.log(f"Test Recall:     {test_metrics[2]*100:>0.4f}%")
    logger.log(f"Test F1:         {test_metrics[3]*100:>0.4f}%\n")

    logger.log(f'Optimization Score:  {final_score*100:>0.5f}%\n\n')


def get_metric_to_follow(model):
    if isinstance(model, Lawin):
        return 3
    return 0
