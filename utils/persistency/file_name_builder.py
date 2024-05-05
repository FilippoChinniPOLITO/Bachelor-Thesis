import os


def file_name_builder(root_folder, folder_num, type_file_name, extension):
    folder_num_path = os.path.join(root_folder, str(folder_num))

    if os.path.exists(folder_num_path):
        if any((type_file_name in file) for file in os.listdir(folder_num_path)):
            raise Exception(f"File {type_file_name} already exists inside {folder_num_path}")
    else:
        os.makedirs(folder_num_path)

    file_path = os.path.join(folder_num_path, f"{type_file_name}_{folder_num}.{extension}")

    return file_path


def folder_exists_check(root_folder, folder_num, type_file_name):
    folder_num_path = os.path.join(root_folder, str(folder_num))

    if os.path.exists(folder_num_path) and (any((type_file_name in file) for file in os.listdir(folder_num_path))):
        raise Exception(f"File {type_file_name} already exists inside {folder_num_path}")
