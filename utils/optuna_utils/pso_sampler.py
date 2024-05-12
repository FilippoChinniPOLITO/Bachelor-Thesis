from typing import Sequence

import numpy as np
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial, TrialState


class PSOSampler(BaseSampler):
    def __init__(self, num_particles: int, max_generations: int):
        super().__init__()
        self.num_particles = num_particles
        self.max_generations = max_generations

        self.hps_bounds = []
        self.bounds = None
        self.swarm = None

        self.global_best_score = -np.inf
        self.global_best_position = None

        self._search_space = IntersectionSearchSpace(include_pruned=True)
        self._n_startup_trials = num_particles
        self._random_sampler = RandomSampler()   # For the first n_startup_trials

    def before_trial(self, study: Study, trial: FrozenTrial):
        print("Before Trial") # TODO: Remove Debugging
        # if self.max_generations is None:
        #     self.max_generations = self.n_total_trials // self.num_particles

        current_gen = (trial.number // self.num_particles) + 1
        particle_id = (trial.number % self.num_particles) + 1

        trial.system_attrs['generation'] = current_gen
        trial.system_attrs['particle'] = particle_id
        print("User Attrs: ", trial.user_attrs) # TODO: Remove Debugging

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial):
        print("Infer Relative Search Space") # TODO: Remove Debugging
        return self._search_space.calculate(study)

    def sample_relative(self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]):
        print("Sample Relative") # TODO: Remove Debugging
        print("Search Spacee: ", search_space) # TODO: Remove Debugging
        print("Params: ", trial.params) # TODO: Remove Debugging

        if search_space == {}:
            print("1° IF: Search Space is Empty") # TODO: Remove Debugging
            return {}

        # states = (TrialState.COMPLETE, TrialState.PRUNED)
        # trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        # if len(trials) < self.n_startup_trials:
        #     print("2° IF") # TODO: Remove Debugging
        #     return {}

        if self.swarm is None:
            self._init_swarm(search_space)

        return self._sample(trial)

    def sample_independent(self, study: Study, trial: FrozenTrial, param_name: str, param_distribution: BaseDistribution):
        print("Sample Independent") # TODO: Remove Debugging
        print("Param Name: ", param_name) # TODO: Remove Debugging
        print("Param Distribution: ", param_distribution) # TODO: Remove Debugging
        print("----------------------------------------------------") # TODO: Remove Debugging
        if trial.number < self._n_startup_trials:
            return self._random_sampler.sample_independent(study, trial, param_name, param_distribution)
        raise NotImplementedError('Independent sampling is not supported for PSO.')

    def _sample(self, trial: FrozenTrial):
        print("Sample") # TODO: Remove Debugging
        particle = self.swarm[trial.system_attrs['particle']]

        hyperparameters = {}
        categorical_hps = {}
        for position, hp in zip(particle.position, self.hps_bounds):
            if isinstance(hp.distribution, CategoricalDistribution):
                if hp.categorical_id not in categorical_hps:
                    categorical_hps[hp.categorical_id] = []
                categorical_hps[hp.categorical_id].append((hp.name, position))
            elif isinstance(hp.distribution, IntDistribution):
                print("Int Distribution: ", hp.name, hp.distribution) # TODO: Remove Debugging
                print("Position: ", position) # TODO: Remove Debugging
                print("Internal Repr: ", hp.distribution.to_internal_repr(position)) # TODO: Remove Debugging
                print("External Repr: ", hp.distribution.to_external_repr(position)) # TODO: Remove Debugging
                param_value_in_internal_repr = hp.distribution.to_internal_repr(position) # TODO: Remove Debugging
                print("Contains int repr: ", hp.distribution._contains(param_value_in_internal_repr)) # TODO: Remove Debugging
                hyperparameters[hp.name] = hp.round_to_nearest_low_multiple(position) # TODO: prova di cambio da round() a int()
            elif isinstance(hp.distribution, FloatDistribution):
                hyperparameters[hp.name] = position
            else:
                raise ValueError('Invalid distribution type')

        print("Categorical HPS: ", categorical_hps) # TODO: Remove Debugging
        for key, value in categorical_hps.items():
            index = np.argmax([t[1] for t in value])
            name = value[index][0]
            hyperparameters[key] = name

        print("END OF SAMPLE(): Hyperparameters: ", hyperparameters) # TODO: Remove Debugging

        return hyperparameters

    def _init_swarm(self, search_space: dict[str, BaseDistribution]):
        print("Init Swarm") # TODO: Remove Debugging
        for name, distribution in search_space.items():
            print("Name: ", name) # TODO: Remove Debugging
            print("Distribution Type: ", type(distribution)) # TODO: Remove Debugging
            print("Distribution: ", distribution) # TODO: Remove Debugging
            if isinstance(distribution, CategoricalDistribution):
                for choice in distribution.choices:
                    self.hps_bounds.append(_HPBound(choice, distribution, name))
            else:
                self.hps_bounds.append(_HPBound(name, distribution))

        self.bounds = np.array([[hp.low, hp.high] for hp in self.hps_bounds])

        self.swarm = {i+1: Particle(self.bounds, particle_id=i+1) for i in range(self.num_particles)}

    def _inertia_factor_update(self, current_iter: int):
        return (0.4 * ((current_iter - self.max_generations) / (self.max_generations ** 2))) + 0.4

    def _cognitive_factor_update(self, current_iter: int):
        return (-3 * (current_iter / self.max_generations)) + 3.5

    def _social_factor_update(self, current_iter: int):
        return (3 * (current_iter / self.max_generations)) + 0.5

    def after_trial(self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None):
        print("After Trial") # TODO: Remove Debugging
        current_gen = trial.system_attrs['generation']
        particle_id = trial.system_attrs['particle']

        print("Current Gen: ", current_gen) # TODO: Remove Debugging
        print("Particle ID: ", particle_id) # TODO: Remove Debugging
        print("System Attrs: ", trial.system_attrs) # TODO: Remove Debugging
        print("Params: ", trial.params) # TODO: Remove Debugging

        if self.swarm is None:
            return

        particle = self.swarm[particle_id]

        fitness = values[0]

        if fitness > particle.personal_best_score:
            particle.personal_best_score = fitness
            particle.personal_best_position = np.copy(particle.position)

        if fitness > self.global_best_score:
            self.global_best_score = fitness
            self.global_best_position = np.copy(particle.position)

        w = self._inertia_factor_update(current_iter=current_gen-1)
        c1 = self._cognitive_factor_update(current_iter=current_gen-1)
        c2 = self._social_factor_update(current_iter=current_gen-1)

        particle.update_velocity(self.global_best_position, w, c1, c2)
        particle.update_position(self.bounds)


class _HPBound:
    def __init__(self, name: str, distribution: BaseDistribution, categorical_id=None):
        self.name = name
        self.distribution = distribution

        if isinstance(distribution, CategoricalDistribution):
            if categorical_id is None:
                raise ValueError('Categorical ID must be provided for CategoricalDistribution')
            self.categorical_id = categorical_id
            self.low = 0
            self.high = 1
        elif isinstance(distribution, (IntDistribution, FloatDistribution)):
            self.low = distribution.low
            self.high = distribution.high
        else:
            raise ValueError('Invalid distribution type')

    def round_to_nearest_low_multiple(self, n):
        if isinstance(self.distribution, IntDistribution):
            return int((((n - self.low) // self.distribution.step) * self.distribution.step) + self.low)


class Particle:
    def __init__(self, bounds, particle_id: int):
        self.particle_id = particle_id

        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1])
        self.velocity = np.random.uniform(-1, 1, size=bounds.shape[0])
        self.personal_best_position = np.copy(self.position)
        self.personal_best_score = -np.inf

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.uniform(0, 1.5, size=self.velocity.shape)
        r2 = np.random.uniform(0, 1.5, size=self.velocity.shape)

        cognitive_velocity = c1 * r1 * (self.personal_best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[:, 0], bounds[:, 1])

