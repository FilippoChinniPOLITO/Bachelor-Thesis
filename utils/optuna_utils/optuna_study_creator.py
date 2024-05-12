import optuna
from optuna.storages import RDBStorage

from utils.persistency.file_name_builder import folder_exists_check, file_name_builder, file_path_builder


class OptunaStudyCreator:
    def __init__(self, experiment_name, path_db, session_num):
        self.experiment_name = experiment_name
        self.path_db = path_db
        self.session_num = session_num

    def __call__(self, study_name, storage, direction, sampler, pruner=None):
        db_study_name = f'study_{study_name}_{self.session_num}'
        storage_obj = self._build_storage(db_study_name, storage)

        return optuna.create_study(study_name=db_study_name,
                                   storage=storage_obj,
                                   direction=direction,
                                   sampler=sampler,
                                   pruner=pruner)

    def _build_storage(self, db_study_name, storage):
        if storage:
            try:
                folder_exists_check(self.path_db, '', f'{self.experiment_name}')
                db_file = file_name_builder(self.path_db, '', f'{self.experiment_name}', 'db')
            except Exception:
                db_file = file_path_builder(self.path_db, f'{self.experiment_name}', '', 'db')

            storage_url = f'sqlite:///{db_file}'
            storage_obj = RDBStorage(url=storage_url, engine_kwargs={"connect_args": {"timeout": 15}})

            try:
                storage_obj.get_study_id_from_name(db_study_name)
                raise ValueError(f"Study {db_study_name} already exists in the database.")
            except KeyError:
                pass
        else:
            storage_obj = None

        return storage_obj

    def load_study_from_db(self, study_name, session):
        db_study_name = f'study_{study_name}_{session}'

        db_file = file_path_builder(self.path_db, f'{self.experiment_name}', '', 'db')
        storage_url = f'sqlite:///{db_file}'

        study = optuna.load_study(study_name=db_study_name, storage=storage_url)
        return study
