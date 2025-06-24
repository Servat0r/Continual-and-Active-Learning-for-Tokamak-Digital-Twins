# Utils for Experiments
from typing import Literal
import shutil
import os
from .logging import LoggingConfiguration
from ..database import *
from ..utils.misc import stdout_debug_print


def __cleanup(what: Literal['aborted', 'tests', 'all'], db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    stdout_debug_print(f"Cleaning up {what} experiments ...", color='red')
    if what == 'aborted':
        count, exp_dicts = db.cleanup_aborted_experiments(targets=targets)
    elif what == 'tests':
        count, exp_dicts = db.cleanup_tests(targets=targets)
    elif what == 'all':
        if targets:
            count, exp_dicts = db.delete_by_id(Experiment, record_ids=targets)
        else:
            raise ValueError(f"Cannot delete with option \"all\" without \"targets\" specified")
    stdout_debug_print(exp_dicts, color='green')
    succ = 0 # Successfully deleted experiments
    if count > 0:
        for exp_dict in exp_dicts:
            logging_config = LoggingConfiguration.from_experiment(exp_dict, db)
            proceed = input(f"Cleaning up experiment {exp_dict['name']}: proceed(y/n)? ")
            if proceed.lower() == 'y':
                for task_id in range(exp_dict['num_tasks']):
                    try:
                        log_folder = logging_config.get_log_folder(mode='new', task_id=task_id)
                        try:
                            shutil.rmtree(log_folder)
                            print(f"Deleted folder and all contents: {log_folder}")
                            if task_id == 0: succ += 1
                        except Exception as e:
                            print(f"Error deleting {log_folder}: {e}")
                    except ValueError: # Finished deleting
                        break
                log_folder = logging_config.get_log_folder(mode='new', suffix=False)
                try:
                    shutil.rmtree(log_folder)
                    print(f"Deleted folder and all contents: {log_folder}")
                    if task_id == 0: succ += 1
                except Exception as e:
                    print(f"Error deleting {log_folder}: {e}")
    print(f"Deleted {succ}/{count} configurations ({succ/count * 100:.2f}%)")
    return succ


def cleanup_aborted_experiments(db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    return __cleanup('aborted', db, targets)

def cleanup_tests(db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    return __cleanup('tests', db, targets)

def cleanup_all(db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    return __cleanup('all', db, targets)
