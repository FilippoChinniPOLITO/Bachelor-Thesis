from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class Particle:
    def __init__(self, bounds, particle_id):
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


class PSO:
    def __init__(self, objective_fn, hps_bounds, num_particles, n_iter):
        self.objective_fn = objective_fn

        self.hps = hps_bounds
        self.bounds = np.array(list(self.hps.values()))
        self.hps = {key: i for i, key in enumerate(self.hps.keys())}

        self.num_particles = num_particles
        self.n_iter = n_iter

        self.trials_list = []
        self.best_trial = None

    def optimize(self, n_jobs=1, logger=None):
        swarm = [Particle(self.bounds, particle_id=_) for _ in range(self.num_particles)]
        global_best_score = -np.inf
        global_best_position = None

        for i in range(self.n_iter):
            for particle in swarm:
                current_trial = PSOTrial(particle_id=particle.particle_id,
                                         generation=i+1,
                                         hyperparameters=self.position_to_hps_map(particle.position))
                fitness = self.objective_fn(current_trial, logger)
                current_trial.complete_trail(score=fitness, state='COMPLETE')
                self.trials_list.append(current_trial)

                if fitness > particle.personal_best_score:
                    particle.personal_best_score = fitness
                    particle.personal_best_position = particle.position

                if fitness > global_best_score:
                    global_best_score = fitness
                    global_best_position = particle.position
                    self.best_trial = current_trial

            w = self.inertia_factor_update(current_iter=i)
            c1 = self.cognitive_factor_update(current_iter=i)
            c2 = self.social_factor_update(current_iter=i)
            for particle in swarm:
                particle.update_velocity(global_best_position, w, c1, c2)
                particle.update_position(self.bounds)

        return global_best_position, global_best_score

    def inertia_factor_update(self, current_iter):
        return (0.4 * ((current_iter - self.n_iter) / (self.n_iter ** 2))) + 0.4

    def cognitive_factor_update(self, current_iter):
        return (-3 * (current_iter / self.n_iter)) + 3.5

    def social_factor_update(self, current_iter):
        return (3 * (current_iter / self.n_iter)) + 0.5

    def position_to_hps_map(self, position):
        return {key: position[self.hps[key]] for key in self.hps.keys()}

    def trials_dataframe(self, attrs=None):
        if attrs is not None:
            return pd.DataFrame([trial.to_dict() for trial in self.trials_list])[attrs]
        return pd.DataFrame([trial.to_dict() for trial in self.trials_list])


class PSOTrial:
    def __init__(self, particle_id, generation, hyperparameters):
        self.particle_id = particle_id
        self.generation = generation
        self.hyperparameters = hyperparameters

        self.datetime_start = datetime.now()

        self.score = None
        self.state = None
        self.duration = None
        self.datetime_complete = None

        self.user_attrs = dict()

    def complete_trail(self, score, state):
        self.score = score
        self.state = state
        self.datetime_complete = datetime.now()
        self.duration = self.datetime_complete - self.datetime_start

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def to_dict(self):
        return {
            'generation': self.generation,
            'particle_id': self.particle_id,
            'score': self.score,
            **{f'user_attrs_{key}': self.user_attrs[key] for key in self.user_attrs},
            'state': self.state,
            **{f'hp_{key}': self.hyperparameters[key] for key in self.hyperparameters},
            'duration': self.duration,
            'datetime_start': self.datetime_start,
            'datetime_complete': self.datetime_complete,
        }


