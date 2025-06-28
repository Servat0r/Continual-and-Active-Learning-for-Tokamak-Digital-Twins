# Rapid Cleanup
import shutil

from src.database import *
from src.utils.logging import LoggingConfiguration


def cleanup_last_experiment(db: SecureMLExperimentDB) -> bool:
    exp_id = db[Experiment][-1]['id']
    print(f"Cleaning up experiment #{exp_id}")
    exp_dict = db.delete_one_by_id(Experiment, exp_id)
    if exp_dict is None:
        print(f"Failed to delete experiment #{exp_id}")
        return False
    logging_config = LoggingConfiguration.from_experiment(exp_dict, db)
    assert logging_config is not None, f"Failed to build {LoggingConfiguration.__name__} from \"{exp_dict}\""
    for task_id in range(exp_dict['num_tasks']):
        try:
            log_folder = logging_config.get_log_folder(mode='new', task_id=task_id)
            try:
                shutil.rmtree(log_folder)
                print(f"Deleted folder and all contents: {log_folder}")
                if task_id == 0: succ += 1
            except Exception as e:
                print(f"Error deleting {log_folder}: {e}")
        except Exception:
            print(f"Error when deleting folders: {e}")
            return False
    log_folder = logging_config.get_log_folder(mode='new', suffix=False)
    try:
        shutil.rmtree(log_folder)
        print(f"Deleted folder and all contents: {log_folder}")
        if task_id == 0: succ += 1
        return True
    except Exception as e:
        print(f"Error deleting {log_folder}: {e}")
        return False


if __name__ == '__main__':
    db = get_db()
    print(cleanup_last_experiment(db))
