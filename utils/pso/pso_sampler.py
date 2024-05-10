import numpy as np
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import IntDistribution, UniformDistribution, CategoricalDistribution, LogUniformDistribution
from optuna.samplers import BaseSampler, intersection_search_space, RandomSampler
from optuna.trial import FrozenTrial, TrialState

from utils.pso.PSO import Particle


class PSOSampler(BaseSampler):
    def __init__(self, num_particles, max_generations, n_startup_trials=8):
        super().__init__()
        self.num_particles = num_particles
        self.max_generations = max_generations
        self.n_startup_trials = n_startup_trials
        self.random_sampler = RandomSampler()   # For the first n_startup_trials

        self.max_generations = None

        self.hps_bounds = []
        self.bounds = None
        self.swarm = None

        self.global_best_score = -np.inf
        self.global_best_position = None

    def before_trial(self, study: Study, trial: FrozenTrial):
        # if self.max_generations is None:
        #     self.max_generations = self.n_total_trials // self.num_particles

        current_gen = (trial.number // self.num_particles) + 1
        particle_id = (trial.number % self.num_particles) + 1

        trial.set_system_attr('Generation', current_gen)
        trial.set_system_attr('Particle', particle_id)

    def infer_relative_search_space(self, study, trial):
        return intersection_search_space(study)

    def sample_relative(self, study, trial, search_space):
        print(search_space) # TODO: Remove Debugging
        print(trial.params) # TODO: Remove Debugging

        if search_space == {}:
            print("1° IF: Search Space is Empty") # TODO: Remove Debugging
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        if len(trials) < self._n_startup_trials:
            print("2° IF") # TODO: Remove Debugging
            return {}

        if self.swarm is None:
            self._init_swarm(search_space)
        return self._sample(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        print("Sample Independent") # TODO: Remove Debugging
        if trial.number < self.n_startup_trials:
            return self.random_sampler.sample_independent(study, trial, param_name, param_distribution)
        raise NotImplementedError('Independent sampling is not supported.')

    def _sample(self, study, trial, search_space):
        particle = self.swarm[trial.system_attrs.get('Particle')]

        hyperparameters = {}
        categorical_hps = {}
        for position, hp in zip(particle.position, self.hps_bounds):
            if isinstance(hp.distribution, CategoricalDistribution):
                if hp.categorical_id not in categorical_hps:
                    categorical_hps[hp.categorical_id] = []
                categorical_hps[hp.categorical_id].append(position)
            elif isinstance(hp.distribution, (IntDistribution, UniformDistribution, LogUniformDistribution)):
                hyperparameters[hp.name] = position
            else:
                raise ValueError('Invalid distribution type')

        hyperparameters.update({key: np.argmax(np.array(value)) for key, value in categorical_hps.items()})

        return hyperparameters

    def _init_swarm(self, search_space):
        for name, distribution in search_space.items():
            if isinstance(distribution, CategoricalDistribution):
                for choice in distribution.choices:
                    self.hps_bounds.append(_HPBound(choice, distribution, name))
            else:
                self.hps_bounds.append(_HPBound(name, distribution))

        self.bounds = np.array([[hp.low, hp.high] for hp in self.hps_bounds])
        print(self.bounds) # TODO: Remove Debugging
        print(self.bounds.shape) # TODO: Remove Debugging

        self.swarm = {i+1: Particle(self.bounds, particle_id=i+1) for i in range(self.num_particles)}

    def inertia_factor_update(self, current_iter):
        return (0.4 * ((current_iter - self.max_generations) / (self.max_generations ** 2))) + 0.4

    def cognitive_factor_update(self, current_iter):
        return (-3 * (current_iter / self.max_generations)) + 3.5

    def social_factor_update(self, current_iter):
        return (3 * (current_iter / self.max_generations)) + 0.5

    def after_trial(self, study: Study, trial: FrozenTrial, state: TrialState, values):
        current_gen = trial.system_attrs.get('Generation')
        particle_id = trial.system_attrs.get('Particle')

        particle = self.swarm[particle_id]

        fitness = values[0]

        if fitness > particle.personal_best_score:
            particle.personal_best_score = fitness
            particle.personal_best_position = np.copy(particle.position)

        if fitness > self.global_best_score:
            self.global_best_score = fitness
            self.global_best_position = np.copy(particle.position)

        w = self.inertia_factor_update(current_iter=current_gen-1)
        c1 = self.cognitive_factor_update(current_iter=current_gen-1)
        c2 = self.social_factor_update(current_iter=current_gen-1)

        particle.update_velocity(self.global_best_position, w, c1, c2)
        particle.update_position(self.bounds)


class _HPBound:
    def __init__(self, name, distribution: BaseDistribution, categorical_id=None):
        self.name = name
        self.distribution = distribution

        if isinstance(distribution, CategoricalDistribution):
            if categorical_id is None:
                raise ValueError('Categorical ID must be provided for CategoricalDistribution')
            self.categorical_id = categorical_id
            self.low = 0
            self.high = 1
        elif isinstance(distribution, (IntDistribution, UniformDistribution, LogUniformDistribution)):
            self.low = distribution.low
            self.high = distribution.high
        else:
            raise ValueError('Invalid distribution type')


