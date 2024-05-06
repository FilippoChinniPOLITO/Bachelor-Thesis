from utils.persistency.file_name_builder import folder_exists_check, file_name_builder
from utils.persistency.logger import Logger
from utils.pso.PSO import PSO

ATTRS = ('generation', 'particle_id', 'hp_num_hidden_layer', 'score', 'user_attrs_epochs', 'user_attrs_network', 'user_attr_accuracy', 'user_attr_precision', 'user_attr_recall', 'user_attr_f1', 'state', 'duration', 'hp_hidden_layer_n1_size', 'hp_hidden_layer_n2_size', 'hp_hidden_layer_n3_size', 'hp_learning_rate' 'datetime_start', 'datetime_complete')


class PSORunner:
    def __init__(self, path_csv, path_txt, session_num, n_jobs, metric_to_follow, attrs=ATTRS):
        self.path_csv = path_csv
        self.path_txt = path_txt
        self.session_num = session_num
        self.n_jobs = n_jobs
        self.metric_to_follow = metric_to_follow
        self.attrs = attrs

    def __call__(self, pso_study: PSO, study_str: str):
        # Init Logger
        folder_exists_check(self.path_csv, self.session_num, f'df_{study_str}')
        folder_exists_check(self.path_txt, self.session_num, f'log_{study_str}')
        logger_study = Logger(file_name_builder(self.path_txt, self.session_num, f'log_{study_str}', 'txt'))

        # Run Optimization
        try:
            pso_study.optimize(logger=logger_study)
        except Exception as e:
            logger_study.err(e)
            return None

        # Log Best Results
        logger_study.log(f"\n\nBEST TRIAL RESULTS:\n\n")
        logger_study.log(f"Best trial Gen n°:       {pso_study.best_trial.generation}")
        logger_study.log(f"Best trial Particle n°:  {pso_study.best_trial.particle_id}")
        logger_study.log(f"Best trial {self.metric_to_follow}-score:    {pso_study.best_trial.user_attrs[self.metric_to_follow]}")
        logger_study.log(f"Best score:              {pso_study.best_trial.score}")
        logger_study.log(f"Best hyperparameters:    {pso_study.best_trial.hyperparameters}")

        logger_study.end_log()

        # Save DataFrame
        df_study = pso_study.trials_dataframe(attrs=self.attrs)
        df_study.to_csv(file_name_builder(self.path_csv, self.session_num, f'df_{study_str}', 'csv'))

        return df_study

