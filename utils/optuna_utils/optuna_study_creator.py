import optuna
from optuna.storages import RDBStorage
from sqlalchemy import create_engine, text

from utils.persistency.file_name_builder import folder_exists_check, file_name_builder, file_path_builder


class OptunaStudyCreator:
    def __init__(self, experiment_name, path_db, session_num, use_storage=False):
        self.experiment_name = experiment_name
        self.path_db = path_db
        self.session_num = session_num

        if use_storage:
            self._build_storage()

    def __call__(self, study_name, direction, sampler, pruner=None):
        db_study_name = f'study_{study_name}_{self.session_num}'
        self._check_study_exists(db_study_name)

        return optuna.create_study(study_name=db_study_name,
                                   storage=self.storage_obj,
                                   direction=direction,
                                   sampler=sampler,
                                   pruner=pruner)

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
        db_file = file_path_builder(self.path_db, f'{self.experiment_name}', '', 'db')
        storage_url = f'sqlite:///{db_file}'

        engine = create_engine(storage_url)
        with engine.connect() as connection:
            # Delete from studies table
            connection.execute(text(f"DELETE FROM studies WHERE study_name = :sname"), {"sname": db_study_name})
            # Delete from trial_params table
            connection.execute(text(f"DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id NOT IN (SELECT study_id FROM studies))"))
            # Delete from trial_values table
            connection.execute(text(f"DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id NOT IN (SELECT study_id FROM studies))"))
            # Delete from trial_intermediate_values table
            connection.execute(text(f"DELETE FROM trial_intermediate_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id NOT IN (SELECT study_id FROM studies))"))
            # Delete from trials table
            connection.execute(text(f"DELETE FROM trials WHERE study_id NOT IN (SELECT study_id FROM studies)"))
