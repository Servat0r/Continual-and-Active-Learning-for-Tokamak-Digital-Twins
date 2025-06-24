# Interface of SecureMLExperimentDB
from sqlalchemy.orm import Session
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Type, Tuple, Iterable
from abc import ABC, abstractmethod

from .utils import *
from .orms import *


class BaseMLExperimentDB(ABC):

    @abstractmethod
    def create_tables(self) -> None:
        pass
    
    @contextmanager
    @abstractmethod
    def get_session(self) -> Session:
        """
        Context manager for database sessions with proper error handling.
        Security: Ensures transactions are properly committed or rolled back.
        """
        pass
    
    @abstractmethod
    def _build_query_conditions(self, query, model_class: Type[TOrm], conditions: Dict) -> Any:
        """
        Build query conditions with security validation.
        Security: Validates operators and prevents SQL injection.
        """
        pass
    
    @abstractmethod
    def create(self, record) -> int:
        """Create a new record with validation."""
        pass
    
    @abstractmethod
    def get_one_by_id(self, model_class: Type[TOrm], record_id: int, as_dict: bool = False) -> Optional[Dict | TOrm]:
        """Read a single record by ID."""
        pass
    
    @abstractmethod
    def get_first(
            self, model_class: Type[TOrm], conditions: Dict, as_dict: bool = False
    ) -> List[Dict]:
        pass
    
    @abstractmethod
    def get(
            self, model_class: Type[TOrm], conditions: Dict, 
            limit: int = 1000, offset: int = 0, as_dict: bool = False
    ) -> List[Dict]:
        """Read records matching conditions with security validation."""
        pass
    
    @abstractmethod
    def update_one_by_id(self, record: TOrm, data: Dict) -> Optional[TOrm]:
        """Update a record with validation."""
        pass
    
    @abstractmethod
    def update_by_id(self, records: List[TOrm], data: List[Dict]) -> Tuple[List[TOrm], List[TOrm]]:
        """Update a list of records."""
        pass
    
    @abstractmethod
    def update_where(
        self, model_class: Type[TOrm], conditions: Dict, data: List[Dict], fields: Optional[Iterable[str]] = None
    ):
        """Update a list of records filtering by given conditions."""
        pass

    @abstractmethod
    def delete_one_by_id(self, record: TOrm) -> Optional[TOrm]:
        """Delete a record."""
        pass
    
    @abstractmethod
    def delete_by_id(self, records: List[TOrm]) -> Tuple[List[TOrm], List[TOrm]]:
        """Delete a list of records."""
        pass

    @abstractmethod
    def delete_where(self, model_class: Type[TOrm], conditions: Dict, fields: Optional[Iterable[str]] = None) -> Tuple[int, List[dict]]:
        """Delete a list of records according to conditions."""
        pass
    
    @abstractmethod
    def count_records(self, model_class: Type[TOrm], conditions: Optional[Dict] = None) -> int:
        """Count records matching conditions."""
        pass
    
    @abstractmethod
    def create_experiment(self, record: Experiment, max_attempts: int = 10) -> Tuple[int, str]:
        """
        Create an experiment with automatic unique name generation.
        Returns:
            Tuple of (experiment_id, final_experiment_name)
        """
        pass
    
    @abstractmethod
    def read_experiments_with_details(
        self, conditions: Optional[Dict] = None,
        limit: int = 1000, offset: int = 0
    ) -> List[Dict]:
        """
        Read experiments with all related configuration details.
        Uses eager loading for optimal performance.
        """
        pass
    
    @abstractmethod
    def search_experiments_advanced(
        self, general_conditions: Optional[Dict] = None,
        scenario_conditions: Optional[Dict] = None,
        architecture_conditions: Optional[Dict] = None,
        experiment_conditions: Optional[Dict] = None,
        limit: int = 1000, offset: int = 0
    ) -> List[Dict]:
        """
        Advanced search across related tables with security validation.
        """
        pass
    
    @abstractmethod
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive experiment statistics."""
        pass
    
    @abstractmethod
    def cleanup_aborted_experiments(self) -> Tuple[int, List[Tuple[int, str]]]:
        """
        Removes data for all experiments with status == "aborted".

        Returns:
            List of tuples (id, name) to be used for cleaning up directories.
        """
        pass
    
    @abstractmethod
    def set_init_to_pending(self, ids: List[int]) -> Tuple[int, List[Experiment]]:
        """
        Sets the status of all these experiments from "init" to "pending".

        Returns:
            List of successfully updated experiments as objects.
        """
        pass
    
    @abstractmethod
    def set_pending_to_running(self, ids: List[int]):
        """
        Sets the status of all these experiments from "pending" to "running".

        Returns:
            List of successfully updated experiments as objects.
        """
        pass
    
    @abstractmethod
    def set_any_to_aborted(self, ids: List[int]):
        """
        Sets the status of all these experiments from any status to "aborted".

        Returns:
            List of successfully updated experiments as objects.
        """
        pass
    
    @abstractmethod
    def set_running_to_finished(self, ids: List[int]):
        """
        Sets the status of all these experiments from "running" to "finished".

        Returns:
            List of successfully updated experiments as objects.
        """
        pass

    @abstractmethod
    def cleanup_orphaned_configs(self) -> Dict[str, int]:
        """
        Remove configuration records that are not referenced by any experiment.
        
        Returns:
            Dictionary with count of deleted records per table
        """
        pass
    
    @abstractmethod
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database structure and content."""
        pass


__all__ = ['BaseMLExperimentDB']
