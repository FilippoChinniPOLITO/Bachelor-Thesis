from torch import cuda

from experiments.PSO_experiment.backend.PSO import PSOTrial
from experiments.weed_mapping_experiment.backend.model.lawin_model import Lawin
from utils.training.eval_step import eval_step
from utils.training.train_step import train_step


def pso_full_train_loop(max_epochs, train_loader, val_loader, test_loader, model, loss_fn, optimizer, early_stopper, regularizer, logger, trial: PSOTrial):

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
                               logger=logger, trial=trial)

        # Check Early-Stopping Step
        if early_stopper(score=optim_score, model=model):
            early_stopping_reporting(logger=logger, trial=trial)
            break

        # Pruning Step
        pso_value = pruning_step(val_metrics=val_metrics_processed,
                                 intermediate_score=optim_score, epoch_index=epoch_index,
                                 logger=logger, trial=trial)
        if pso_value is not None:
            return pso_value

    # Complete Training
    completing_reporting(logger=logger, trial=trial)
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
                               logger=logger, trial=trial)

    # Empty Cache
    # cuda.empty_cache()

    return final_optim_score


def intermediate_reporting(val_loss, val_metrics, intermediate_score, epoch_index, logger, trial):
    trial.set_user_attr(key='epochs', value=epoch_index+1)

    logger.log(f"\nTrial Gen n°{trial.generation} - Particle n°{trial.particle_id} - Epoch {epoch_index+1}\n----------------------------------------")
    logger.log(f"Intermediate Avg loss:   {val_loss:>0.4f}")
    logger.log(f"Intermediate Accuracy:   {val_metrics[0]*100:>0.4f}%")
    logger.log(f"Intermediate Precision:  {val_metrics[1]*100:>0.4f}%")
    logger.log(f"Intermediate Recall:     {val_metrics[2]*100:>0.4f}%")
    logger.log(f"Intermediate F1:         {val_metrics[3]*100:>0.4f}%\n")

    logger.log(f'Intermediate Optimization Score (Gen n°{trial.generation} - Particle n°{trial.particle_id}): {intermediate_score*100:>0.4f}%\n')


def early_stopping_reporting(logger, trial):
    logger.log(f"Trial Gen n°{trial.generation} - Particle n°{trial.particle_id} Early Stopped!")


def pruning_step(val_metrics, intermediate_score, epoch_index, logger, trial):
    trial.report(score=intermediate_score, step=epoch_index)
    if (intermediate_score < 0) or (trial.should_prune()):
        trial.set_user_attr(key='accuracy', value=round(val_metrics[0], 4))
        trial.set_user_attr(key='precision', value=round(val_metrics[1], 4))
        trial.set_user_attr(key='recall', value=round(val_metrics[3], 4))
        trial.set_user_attr(key='f1', value=round(val_metrics[4], 4))

        logger.log(f"Trial Gen n°{trial.generation} - Particle n°{trial.particle_id} Pruned!\n\n")

        # cuda.empty_cache()
        trial.prune_trial()
        return intermediate_score


def completing_reporting(logger, trial):
    logger.log(f"Training Gen n°{trial.generation} - Particle n°{trial.particle_id} Complete!\n\n")


def final_evaluation_reporting(test_metrics, final_score, logger, trial):
    trial.set_user_attr(key='accuracy', value=round(test_metrics[0], 4))
    trial.set_user_attr(key='precision', value=round(test_metrics[1], 4))
    trial.set_user_attr(key='recall', value=round(test_metrics[2], 4))
    trial.set_user_attr(key='f1', value=round(test_metrics[3], 4))

    logger.log(f"\nReport Finished Trial:\n----------------------------------------")
    logger.log(f"Trial: Gen n°{trial.generation} - Particle n°{trial.particle_id}")
    logger.log(f"Hyperparameters: {trial.hyperparameters}")

    logger.log(f"Test Accuracy:   {test_metrics[0]*100:>0.4f}%")
    logger.log(f"Test Precision:  {test_metrics[1]*100:>0.4f}%")
    logger.log(f"Test Recall:     {test_metrics[2]*100:>0.4f}%")
    logger.log(f"Test F1:         {test_metrics[3]*100:>0.4f}%\n")

    logger.log(f'Optimization Score:  {final_score*100:>0.5f}%\n\n')


def get_metric_to_follow(model):
    if isinstance(model, Lawin):
        return 3
    return 0
