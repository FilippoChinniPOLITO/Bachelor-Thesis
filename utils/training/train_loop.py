import optuna
from torch import cuda
from optuna import Trial

from utils.pso.PSO import PSOTrial
from utils.training.train_step import train_step
from utils.training.eval_step import eval_step
from utils.training.eval_step import eval_step_weedmapping


def full_train_loop(max_epochs, train_loader, val_loader, test_loader, model, loss_fn, optimizer, early_stopper, regularizer, logger, trial=None):
    # Init for Missing Parameters
    is_optuna = False
    is_pso = False
    if trial is not None:
        if isinstance(trial, Trial):
            is_optuna = True
        elif isinstance(trial, PSOTrial):
            is_pso = True
    if regularizer is None:
        regularizer = mock_regularizer

    # Train Model
    for epoch_index in range(max_epochs):

        # Training Step
        train_step(train_loader, model, loss_fn, optimizer)

        # Intermediate Evaluation Step
        _ = eval_step(val_loader, model, loss_fn)
        val_loss = _[0]
        accuracy_score, precision_score, recall_score, f1_score = (i.item() for i in _[1:])

        # Print Metrics
        if is_optuna:
            trial.set_user_attr(key='epochs', value=epoch_index+1)
            logger.log(f"\nTrial n°{trial.number} - Epoch {epoch_index+1}\n-------------------------------")
        elif is_pso:
            trial.set_user_attr(key='epochs', value=epoch_index+1)
            logger.log(f"\nTrial Gen n°{trial.generation} - Particle n°{trial.particle_id} - Epoch {epoch_index+1}\n-------------------------------")
        else:
            logger.log(f"Training - \nEpoch {epoch_index+1}\n-------------------------------")
        logger.log(f"Intermediate Avg loss:   {val_loss:>0.4f}")
        logger.log(f"Intermediate Accuracy:   {accuracy_score*100:>0.4f}%")
        logger.log(f"Intermediate Precision:  {precision_score*100:>0.4f}%")
        logger.log(f"Intermediate Recall:     {recall_score*100:>0.4f}%")
        logger.log(f"Intermediate F1:         {f1_score*100:>0.4f}%\n")

        # Intermediate Optimization Score
        optim_score = regularizer(score=accuracy_score, network_architecture=model.get_network_architecture())
        if is_optuna:
            logger.log(f'Intermediate Optimization Score (Trial: n°{trial.number}): {optim_score*100:>0.3f}%\n')
        elif is_pso:
            logger.log(f'Intermediate Optimization Score (Gen n°{trial.generation} - Particle: n°{trial.particle_id}): {optim_score*100:>0.3f}%\n')
        else:
            logger.log(f'Intermediate Optimization Score: {optim_score*100:>0.3f}%\n')

        # Check Early-Stopping Step
        if early_stopper(score=optim_score, model=model):
            model.load_state_dict(early_stopper.get_best_model_params())
            if is_optuna:
                logger.log(f"Trial n°{trial.number} Early Stopped!")
            elif is_pso:
                logger.log(f"Trial Gen n°{trial.generation} - Particle n°{trial.particle_id} Early Stopped!")
            else:
                logger.log(f"Training Early Stopped!")
            break

        # Pruning Step
        if is_optuna:
            trial.report(value=early_stopper.get_best_score(), step=epoch_index)
            if (optim_score < 0) or (trial.should_prune()):
                trial.set_user_attr(key='accuracy', value=accuracy_score)
                trial.set_user_attr(key='precision', value=precision_score)
                trial.set_user_attr(key='recall', value=recall_score)
                trial.set_user_attr(key='f1', value=f1_score)
                logger.log(f"Trial n°{trial.number} Pruned!\n\n")
                cuda.empty_cache()
                raise optuna.TrialPruned()
        if is_pso:
            trial.report(score=early_stopper.get_best_score(), step=epoch_index)
            logger.test(f'ARRIVED IN is_pso\nshould: {trial.should_prune()}')  # TODO Remove test
            if (optim_score < 0) or (trial.should_prune()):
                trial.set_user_attr(key='accuracy', value=accuracy_score)
                trial.set_user_attr(key='precision', value=precision_score)
                trial.set_user_attr(key='recall', value=recall_score)
                trial.set_user_attr(key='f1', value=f1_score)
                logger.log(f"Trial Gen n°{trial.generation} - Particle n°{trial.particle_id} Pruned!\n\n")
                cuda.empty_cache()
                trial.prune_trial()
                return early_stopper.get_best_score()

    if is_optuna:
        logger.log(f"Training n°{trial.number} Complete!\n\n")
    elif is_pso:
        logger.log(f"Training Gen n°{trial.generation} - Particle n°{trial.particle_id} Complete!\n\n")
    else:
        logger.log(f"Training Complete!\n\n")


    # Evaluate Model - Evaluation Metrics
    _ = eval_step(test_loader, model, loss_fn)
    final_accuracy_score, final_precision_score, final_recall_score, final_f1_score = (i.item() for i in _[1:])
    if is_optuna:
        trial.set_user_attr(key='accuracy', value=final_accuracy_score)
        trial.set_user_attr(key='precision', value=final_precision_score)
        trial.set_user_attr(key='recall', value=final_recall_score)
        trial.set_user_attr(key='f1', value=final_f1_score)
        logger.log(f"Trial number: {trial.number}")
        logger.log(f"Hyperparameters: {trial.params}")
    elif is_pso:
        trial.set_user_attr(key='accuracy', value=final_accuracy_score)
        trial.set_user_attr(key='precision', value=final_precision_score)
        trial.set_user_attr(key='recall', value=final_recall_score)
        trial.set_user_attr(key='f1', value=final_f1_score)
        logger.log(f"Trial: Gen n°{trial.generation} - Particle n°{trial.particle_id}")
        logger.log(f"Hyperparameters: {trial.hyperparameters}")
    logger.log(f"Test Accuracy:   {final_accuracy_score*100:>0.4f}%")
    logger.log(f"Test Precision:  {final_precision_score*100:>0.4f}%")
    logger.log(f"Test Recall:     {final_recall_score*100:>0.4f}%")
    logger.log(f"Test F1:         {final_f1_score*100:>0.4f}%\n")

    # Evaluate Model - Optimization Score
    final_optim_score = regularizer(score=final_accuracy_score, network_architecture=model.get_network_architecture())
    logger.log(f'Optimization Score:  {final_optim_score*100:>0.5f}%\n\n')


    # Empty Cache
    cuda.empty_cache()

    return final_optim_score


def full_train_loop_weedmapping(max_epochs, train_loader, val_loader, test_loader, model, backbone_str, loss_fn, optimizer, early_stopper, regularizer, logger, trial=None):
    # Init for Missing Parameters
    is_optuna = False
    is_pso = False
    if trial is not None:
        if isinstance(trial, Trial):
            is_optuna = True
        elif isinstance(trial, PSOTrial):
            is_pso = True
    if regularizer is None:
        regularizer = mock_regularizer

    # Train Model
    for epoch_index in range(max_epochs):

        # Training Step
        train_step(train_loader, model, loss_fn, optimizer)

        # Intermediate Evaluation Step
        _ = eval_step_weedmapping(val_loader, model, loss_fn)
        val_loss = _[0]
        f1_score, precision_score, recall_score = (i.item() for i in _[1:])

        # Print Metrics
        if is_optuna:
            trial.set_user_attr(key='epochs', value=epoch_index+1)
            logger.log(f"\nTrial n°{trial.number} - Epoch {epoch_index+1}\n-------------------------------")
        elif is_pso:
            trial.set_user_attr(key='epochs', value=epoch_index+1)
            logger.log(f"\nTrial Gen n°{trial.generation} - Particle n°{trial.particle_id} - Epoch {epoch_index+1}\n-------------------------------")
        else:
            logger.log(f"Training - \nEpoch {epoch_index+1}\n-------------------------------")
        logger.log(f"Intermediate Avg loss:   {val_loss:>0.4f}")
        logger.log(f"Intermediate F1:         {f1_score*100:>0.4f}%")
        logger.log(f"Intermediate Precision:  {precision_score*100:>0.4f}%")
        logger.log(f"Intermediate Recall:     {recall_score*100:>0.4f}%")

        # Intermediate Optimization Score
        optim_score = regularizer(score=f1_score, backbone_str=backbone_str)
        if is_optuna:
            logger.log(f'Intermediate Optimization Score (Trial: n°{trial.number}): {optim_score*100:>0.3f}%\n')
        elif is_pso:
            logger.log(f'Intermediate Optimization Score (Gen n°{trial.generation} - Particle: n°{trial.particle_id}): {optim_score*100:>0.3f}%\n')
        else:
            logger.log(f'Intermediate Optimization Score: {optim_score*100:>0.3f}%\n')

        # Check Early-Stopping Step
        if early_stopper(score=optim_score, model=model):
            model.load_state_dict(early_stopper.get_best_model_params())
            if is_optuna:
                logger.log(f"Trial n°{trial.number} Early Stopped!")
            elif is_pso:
                logger.log(f"Trial Gen n°{trial.generation} - Particle n°{trial.particle_id} Early Stopped!")
            else:
                logger.log(f"Training Early Stopped!")
            break

        # Pruning Step
        if is_optuna:
            trial.report(value=early_stopper.get_best_score(), step=epoch_index)
            if (optim_score < 0) or (trial.should_prune()):
                trial.set_user_attr(key='f1', value=f1_score)
                trial.set_user_attr(key='precision', value=precision_score)
                trial.set_user_attr(key='recall', value=recall_score)
                logger.log(f"Trial n°{trial.number} Pruned!\n\n")
                cuda.empty_cache()
                raise optuna.TrialPruned()
        if is_pso:
            trial.report(score=early_stopper.get_best_score(), step=epoch_index)
            if (optim_score < 0) or (trial.should_prune()):
                trial.set_user_attr(key='f1', value=f1_score)
                trial.set_user_attr(key='precision', value=precision_score)
                trial.set_user_attr(key='recall', value=recall_score)
                logger.log(f"Trial Gen n°{trial.generation} - Particle n°{trial.particle_id} Pruned!\n\n")
                cuda.empty_cache()
                return early_stopper.get_best_score()

    if is_optuna:
        logger.log(f"Training n°{trial.number} Complete!\n\n")
    elif is_pso:
        logger.log(f"Training Gen n°{trial.generation} - Particle n°{trial.particle_id} Complete!\n\n")
    else:
        logger.log(f"Training Complete!\n\n")


    # Evaluate Model - Evaluation Metrics
    _ = eval_step_weedmapping(test_loader, model, loss_fn)
    final_f1_score, final_precision_score, final_recall_score = (i.item() for i in _[1:])
    if is_optuna:
        trial.set_user_attr(key='f1', value=final_f1_score)
        trial.set_user_attr(key='precision', value=final_precision_score)
        trial.set_user_attr(key='recall', value=final_recall_score)
        logger.log(f"Trial number: {trial.number}")
        logger.log(f"Hyperparameters: {trial.params}")
    elif is_pso:
        trial.set_user_attr(key='f1', value=final_f1_score)
        trial.set_user_attr(key='precision', value=final_precision_score)
        trial.set_user_attr(key='recall', value=final_recall_score)
        logger.log(f"Trial: Gen n°{trial.generation} - Particle n°{trial.particle_id}")
        logger.log(f"Hyperparameters: {trial.hyperparameters}")
    logger.log(f"Test F1:         {final_f1_score*100:>0.4f}%")
    logger.log(f"Test Precision:  {final_precision_score*100:>0.4f}%")
    logger.log(f"Test Recall:     {final_recall_score*100:>0.4f}%")

    # Evaluate Model - Optimization Score
    final_optim_score = regularizer(score=final_f1_score, backbone_str=backbone_str)
    logger.log(f'Optimization Score:  {final_optim_score*100:>0.5f}%\n\n')


    # Empty Cache
    cuda.empty_cache()

    return final_optim_score


def mock_regularizer(**kwargs):
    return kwargs.get('score', None)
