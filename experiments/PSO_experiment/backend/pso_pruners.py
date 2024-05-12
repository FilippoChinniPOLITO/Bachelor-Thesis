import numpy as np

from experiments.PSO_experiment.backend.PSO import PSOTrial


class BasePSOPruner:
    def __init__(self, n_startup_generations):
        self.n_startup_generations = n_startup_generations
        self.is_pruning_active = False

    def active_pruning(self, generation):
        if generation >= self.n_startup_generations:
            self.is_pruning_active = True

    def report(self, score, step):
        raise NotImplementedError

    def should_prune(self, trial: PSOTrial):
        raise NotImplementedError


class PSOMedianPruner(BasePSOPruner):
    def __init__(self, n_startup_generations, n_warmup_steps, interval_steps, min_trials_per_step):
        super().__init__(n_startup_generations)
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.min_trials_per_step = min_trials_per_step

        self.reports_map = {}

    def report(self, score, step):
        if step not in self.reports_map:
            self.reports_map[step] = []
        self.reports_map[step].append(score)

    def should_prune(self, trial: PSOTrial):
        if not self.is_pruning_active:
            return False
        if trial.last_reported_step <= self.n_warmup_steps:
            return False
        if trial.last_reported_step % self.interval_steps != 0:
            return False
        if len(self.reports_map[trial.last_reported_step]) < self.min_trials_per_step:
            return False
        if trial.best_reported_score < np.median(self.reports_map[trial.last_reported_step]):
            return True
        return False

