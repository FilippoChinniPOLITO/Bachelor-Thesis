import optuna

from utils.persistency.file_name_builder import folder_exists_check, file_name_builder


class OptunaStudyCreator:
    def __init__(self, path_db, session_num):
        self.path_db = path_db
        self.session_num = session_num

    def __call__(self, study_name, direction, sampler, pruner=None):
        folder_exists_check(self.path_db, self.session_num, f'storage_{study_name}')
        db_file = file_name_builder(self.path_db, self.session_num, f'storage_{study_name}', 'db')

        storage = f'sqlite:///{db_file}'

        return optuna.create_study(study_name=study_name,
                                   storage=storage,
                                   direction=direction,
                                   sampler=sampler,
                                   pruner=pruner)
