import optuna
from optuna.storages import RDBStorage

from utils.persistency.file_name_builder import folder_exists_check, file_name_builder


class OptunaStudyCreator:
    def __init__(self, path_db, session_num):
        self.path_db = path_db
        self.session_num = session_num

    def __call__(self, study_name, storage, direction, sampler, pruner=None):
        folder_exists_check(self.path_db, self.session_num, f'storage_{study_name}')
        db_file = file_name_builder(self.path_db, self.session_num, f'storage_{study_name}', 'db')

        if storage:
            storage_url = f'sqlite:///{db_file}'
            storage_obj = RDBStorage(url=storage_url, engine_kwargs={"connect_args": {"timeout": 10}})
        else:
            storage_obj = None

        return optuna.create_study(study_name=study_name,
                                   storage=storage_obj,
                                   direction=direction,
                                   sampler=sampler,
                                   pruner=pruner)
