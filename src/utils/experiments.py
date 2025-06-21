# Utils for Experiments
from typing import Literal
import shutil
import os
from .logging import LoggingConfiguration
from ..database import *


def __cleanup(what: Literal['aborted', 'tests', 'all'], db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    if what == 'aborted':
        count, exp_dicts = db.cleanup_aborted_experiments(targets)
    elif what == 'tests':
        count, exp_dicts = db.cleanup_tests(targets)
    elif what == 'all':
        if targets:
            count, exp_dicts = db.delete_records(Experiment, targets)
        else:
            raise ValueError(f"Cannot delete with option \"all\" without \"targets\" specified")
    succ = 0 # Successfully deleted experiments
    if count > 0:
        for exp_dict in exp_dicts:
            logging_config = LoggingConfiguration.from_experiment(exp_dict, db)
            log_folder = logging_config.get_log_folder(mode='new')
            proceed = input(f"Cleaning up experiment at folder {log_folder}: proceed(y/n)? ")
            if proceed == 'y' and os.path.exists(log_folder):
                try:
                    shutil.rmtree(log_folder)
                    print(f"Deleted folder and all contents: {log_folder}")
                    succ += 1
                except Exception as e:
                    print(f"Error deleting {log_folder}: {e}")
    print(f"Deleted {succ}/{count} configuration ({succ/count * 100:.2f}%)")
    return succ


def cleanup_aborted_experiments(db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    return __cleanup('aborted', db, targets)


def cleanup_tests(db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    return __cleanup('tests', db, targets)

def cleanup_all(db: SecureMLExperimentDB, targets: Optional[list[int]] = None) -> int:
    return __cleanup('all', db, targets)
