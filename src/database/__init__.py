from typing import Optional
import os
from .utils import *
from .orms import *
from .interface import *
from .db import *


def get_test_db(test_file: Optional[str] = None) -> SecureMLExperimentDB:
    if test_file is None:
        test_file = os.environ.get("DB_TEST_FILE", DEFAULT_DB_TEST_FILE)
    return SecureMLExperimentDB(f"sqlite:///{test_file}", echo=False, overwrite_db=True, overwrite_consent=False)


def get_db(db_file: Optional[str] = None) -> SecureMLExperimentDB:
    if db_file is None:
        db_file = os.environ.get("DB_FILE", DEFAULT_DB_FILE)
    return SecureMLExperimentDB(f"sqlite:///{db_file}", echo=False, overwrite_db=False, overwrite_consent=True)
