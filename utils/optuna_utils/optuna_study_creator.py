import pickle

import optuna
from optuna.storages import RDBStorage

from utils.persistency.file_name_builder import folder_exists_check, file_name_builder, file_path_builder


class OptunaStudyCreator:
    def __init__(self, experiment_name, path_db, session_num, use_storage=False):
        self.experiment_name = experiment_name
        self.path_db = path_db
        self.session_num = session_num

        self.storage_obj = None

        if use_storage:
            self._build_storage()

    def __call__(self, study_name, direction, sampler, pruner=None, load=False):
        db_study_name = f'study_{study_name}_{self.session_num}'
        final_sampler = sampler
        final_pruner = pruner

        if (not load) and (self.storage_obj is not None):
            self._check_study_exists(db_study_name)

        if load and (self.storage_obj is not None):
            final_sampler = self._load_sampler_from_pickle(study_name)
            final_pruner = self._load_pruner_from_pickle(study_name)

        return optuna.create_study(study_name=db_study_name,
                                   storage=self.storage_obj,
                                   direction=direction,
                                   sampler=final_sampler,
                                   pruner=final_pruner,
                                   load_if_exists=load)

    def _build_storage(self):
        try:
            folder_exists_check(self.path_db, '', f'{self.experiment_name}')
            db_file = file_name_builder(self.path_db, '', f'{self.experiment_name}', 'db')
        except Exception:
            db_file = file_path_builder(self.path_db, f'{self.experiment_name}', '', 'db')

        storage_url = f'sqlite:///{db_file}'
        self.storage_obj = RDBStorage(url=storage_url, engine_kwargs={"connect_args": {"timeout": 30}})

    def _check_study_exists(self, db_study_name):
        try:
            self.storage_obj.get_study_id_from_name(db_study_name)
            raise ValueError(f"Study {db_study_name} already exists in the database.")
        except KeyError:
            pass

    def load_study_from_db(self, study_name, session):
        db_study_name = f'study_{study_name}_{session}'

        db_file = file_path_builder(self.path_db, f'{self.experiment_name}', '', 'db')
        storage_url = f'sqlite:///{db_file}'

        study = optuna.load_study(study_name=db_study_name, storage=storage_url)
        return study

    def copy_study_into_new_db(self, study_name, session, from_db_name):
        db_study_name = f'study_{study_name}_{session}'
        db_file = file_path_builder(self.path_db, f'{from_db_name}', '', 'db')

        optuna.copy_study(from_study_name=db_study_name,
                          to_study_name=db_study_name,
                          from_storage=f'sqlite:///{db_file}',
                          to_storage=self.storage_obj)

    def delete_study_from_db(self, study_name, session):
        db_study_name = f'study_{study_name}_{session}'
        try:
            optuna.delete_study(study_name=db_study_name, storage=self.storage_obj)
            print(f"Study {study_name} deleted from DB.")
        except Exception as e:
            print(f"Failed to delete study {study_name}.")
            print("Error: ", e)

    def _load_sampler_from_pickle(self, study_str):
        path_sampler = self.path_db + '/samplers'
        pickle_file = file_name_builder(path_sampler, self.session_num, f'sampler_{study_str}', 'pkl')

        restored_sampler = pickle.load(open(pickle_file, "rb"))
        return restored_sampler

    def _load_pruner_from_pickle(self, study_str):
        path_pruner = self.path_db + '/pruners'
        pickle_file = file_name_builder(path_pruner, self.session_num, f'pruner_{study_str}', 'pkl')

        restored_pruner = pickle.load(open(pickle_file, "rb"))
        return restored_pruner

