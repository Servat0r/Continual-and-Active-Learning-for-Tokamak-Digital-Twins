from .utils import *
from .orms import *
from .interface import *
from .db import *


def get_test_db(test_file: str = DEFAULT_DB_TEST_FILE) -> SecureMLExperimentDB:
    return SecureMLExperimentDB(f"sqlite:///{test_file}", echo=False, overwrite_db=True, overwrite_consent=False)


def get_db(db_file: str = DEFAULT_DB_FILE) -> SecureMLExperimentDB:
    return SecureMLExperimentDB(f"sqlite:///{db_file}", echo=False, overwrite_db=False, overwrite_consent=True)
