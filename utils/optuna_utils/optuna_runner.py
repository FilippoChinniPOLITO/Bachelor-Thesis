import pickle

from optuna import Study

from utils.persistency.file_name_builder import folder_exists_check, file_name_builder
from utils.persistency.logger import Logger

ATTRS = ('number', 'value', 'user_attrs', 'state', 'params', 'duration', 'datetime_start', 'datetime_complete')


class OptunaRunner:
    def __init__(self, objective_fn, n_jobs, n_trials, path_db, path_csv, path_txt, session_num, metric_to_follow, attrs=ATTRS):
        self.objective_fn = objective_fn
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.path_db = path_db
        self.path_csv = path_csv
        self.path_txt = path_txt
        self.session_num = session_num
        self.metric_to_follow = metric_to_follow
        self.attrs = attrs

    def __call__(self, study: Study, study_str: str, load=False):
        # Init Logger
        if not load:
            folder_exists_check(self.path_csv, self.session_num, f'df_{study_str}')
            folder_exists_check(self.path_txt, self.session_num, f'log_{study_str}')
        logger_study = Logger(file_name_builder(self.path_txt, self.session_num, f'log_{study_str}', 'txt'))

        # Run Optimization
        try:
            study.optimize(lambda trail: self.objective_fn(trail, logger_study), n_trials=self.n_trials, n_jobs=self.n_jobs)
        except Exception as e:
            logger_study.err(e)
            return None
        finally:
            self._save_sampler(study, study_str)
            self._save_pruner(study, study_str)

        # Log Best Results
        logger_study.log(f"\n\nBEST TRIAL RESULTS:\n\n")
        logger_study.log(f"Best trial n°:         {study.best_trial.number}")
        logger_study.log(f"Trial {self.metric_to_follow}-score:  {study.best_trial.user_attrs[self.metric_to_follow]}")
        logger_study.log(f"Best score:            {study.best_trial.value}")
        logger_study.log(f"Best hyperparameters:  {study.best_params}")

        logger_study.end_log()

        # Save DataFrame
        df_study = study.trials_dataframe(attrs=self.attrs)
        df_study.to_csv(file_name_builder(self.path_csv, self.session_num, f'df_{study_str}', 'csv'))

        return df_study

    def _save_sampler(self, study, study_str):
        path_sampler = self.path_db + '/samplers'
        pickle_file = file_name_builder(path_sampler, self.session_num, f'sampler_{study_str}', 'pkl')
        with open(pickle_file, "wb") as fout:
            pickle.dump(study.sampler, fout)
        return pickle_file

    def _save_pruner(self, study, study_str):
        path_pruner = self.path_db + '/pruners'
        pickle_file = file_name_builder(path_pruner, self.session_num, f'pruner_{study_str}', 'pkl')
        with open(pickle_file, "wb") as fout:
            pickle.dump(study.pruner, fout)
        return pickle_file

