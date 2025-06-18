# Generated with Claude 4 Sonnet, with further modifies by Salvatore Correnti
from sqlalchemy import create_engine, func, JSON, Integer, Float, Boolean, cast
from sqlalchemy.orm import sessionmaker, Session, joinedload, undefer
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Tuple
import logging
import os, json

from .utils import *
from .orms import *
from .interface import *

# Configure logging for security monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_FILE = "secure_ml_experiments.db"
DEFAULT_DB_TEST_FILE = "test.db"


# MAIN DATABASE MANAGER CLASS
class SecureMLExperimentDB(BaseMLExperimentDB):
    """
    Secure SQLAlchemy ORM-based database manager for ML experiments.
    
    Security features:
    - SQL injection protection through ORM
    - Input validation and sanitization
    - Proper error handling and logging
    - Transaction management
    - Connection pooling
    - Type safety
    """
    
    def __init__(
            self, db_url: str = "sqlite:///" + DEFAULT_DB_FILE, echo: bool = False,
            overwrite_db: bool = False, overwrite_consent: bool = True
    ):
        """
        Initialize the database manager.
        
        Args:
            db_url: Database URL (SQLite by default for security)
            echo: Whether to echo SQL queries (disable in production)
        """
        assert db_url.startswith('sqlite:///'), f"Invalid Database URL: {db_url}"
        # Security: Use connection pooling and disable echo in production
        self.engine = create_engine(
            db_url,
            echo=echo,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
            # Security: Set reasonable connection limits
            pool_pre_ping=True,
            pool_recycle=3600  # Recycle connections after 1 hour
        )
        
        db_file = db_url[len('sqlite:///'):]
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        if not os.path.exists(db_file):
            self.create_tables()
        elif overwrite_db:
            authorization = 'y'
            if overwrite_consent:
                authorization = input("[WARNING] This method will overwrite previous database: are you sure to proceed (y/n)? ").lower()
            if authorization == 'y':
                try:
                    os.remove(db_file)
                    logger.info(f"Removed existing database file: {db_file}")
                except Exception as e:
                    logger.error(f"Failed to remove database file {db_file}: {e}")
                    raise
                self.create_tables()
        logger.info(f"Database initialized with URL: {db_url}")
    
    def create_tables(self):
        """Create all tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions with proper error handling.
        Security: Ensures transactions are properly committed or rolled back.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            session.close()
    
    def _build_query_conditions(self, query, model_class: Type[TOrm], conditions: Dict) -> Any:
        """
        Build query conditions with simplified JSON field support.
        
        Supports three condition formats:
        1. Simple equality: {'field': value}
        2. Operator tuple: {'field': ('operator', value)}
        3. JSON nested dict: {'json_field': {'key': value, 'key2': ('operator', value)}}
        
        Examples:
            # Regular field conditions
            {'name': 'resnet50', 'id': ('in', [1,2,3])}
            
            # JSON field conditions
            {'parameters': {
                'learning_rate': 0.001,                    # parameters.learning_rate == 0.001
                'optimizer': 'adam',                       # parameters.optimizer == 'adam'
                'epochs': ('gte', 100),                    # parameters.epochs >= 100
                'batch_size': ('in', [32, 64, 128])        # parameters.batch_size in [32,64,128]
            }}
            
            # Mixed conditions
            {
                'name': ('like', 'resnet%'),
                'parameters': {
                    'learning_rate': ('gte', 0.001),
                    'optimizer': ('in', ['adam', 'adamw'])
                },
                'config': {
                    'model_type': 'transformer',
                    'layers': ('gt', 20)
                }
            }
        """
        if not conditions:
            return query
        # Get model fields for validation
        model_fields = {c.name: c for c in model_class.__table__.columns}
        for field_name, condition in conditions.items():
            if field_name not in model_fields:
                raise SecurityError(f"Invalid field name: {field_name}")
            field = model_fields[field_name]
            # Check if this is a JSON field with nested dictionary conditions
            if isinstance(condition, dict) and self._is_json_field(field):
                query = self._apply_json_nested_conditions(query, field, condition)
            # Handle regular field conditions (existing logic)
            elif isinstance(condition, tuple) and len(condition) == 2:
                operator, value = condition
                if operator not in ALLOWED_OPERATORS:
                    raise SecurityError(f"Invalid operator: {operator}")
                # Additional validation for IN operators
                if operator in ['in', 'not_in']:
                    if not isinstance(value, (list, tuple)):
                        raise ValidationError(f"Value for {operator} must be a list or tuple")
                    if len(value) > 100:  # Security: Limit IN clause size
                        raise SecurityError("IN clause too large (max 100 items)")
                query = query.filter(ALLOWED_OPERATORS[operator](field, value))
            else:
                # Simple equality
                query = query.filter(field == condition)
        return query
    
    def _apply_json_nested_conditions(self, query, json_field, nested_conditions: Dict[str, Any]):
        """
        Apply nested JSON conditions from a dictionary structure.
        
        Args:
            query: SQLAlchemy query object
            json_field: The JSON column
            nested_conditions: Dictionary of JSON key conditions
            
        Examples:
            nested_conditions = {
                'learning_rate': 0.001,                    # Simple equality
                'optimizer': 'adam',                       # Simple equality  
                'epochs': ('gte', 100),                    # Operator + value
                'batch_size': ('in', [32, 64, 128]),       # IN operator
                'model_config': ('json_contains', {...}), # JSON operator
            }
        """
        for json_key, json_condition in nested_conditions.items():
            # Security: Validate JSON key
            if not self._is_safe_json_key(json_key):
                raise SecurityError(f"Invalid JSON key: {json_key}")
            
            # Create JSON path accessor
            json_accessor = json_field[json_key]
            
            if isinstance(json_condition, tuple) and len(json_condition) == 2:
                operator, value = json_condition
                
                if operator not in ALLOWED_OPERATORS:
                    raise SecurityError(f"Invalid operator for JSON field: {operator}")
                
                # Handle JSON-specific operators
                if operator.startswith('json_'):
                    query = self._apply_json_specific_operator(query, json_accessor, operator, value)
                else:
                    # Handle regular operators on JSON values with type casting
                    typed_accessor = self._cast_json_value(json_accessor, value)
                    
                    # Additional validation for IN operators
                    if operator in ['in', 'not_in']:
                        if not isinstance(value, (list, tuple)):
                            raise ValidationError(f"Value for JSON {operator} must be a list or tuple")
                        if len(value) > 100:
                            raise SecurityError("JSON IN clause too large (max 100 items)")
                    
                    query = query.filter(ALLOWED_OPERATORS[operator](typed_accessor, value))
            else:
                # Simple equality for JSON field
                typed_accessor = self._cast_json_value(json_accessor, json_condition)
                query = query.filter(typed_accessor == json_condition)
        
        return query

    def _apply_json_specific_operator(self, query, json_accessor, operator: str, value):
        """Apply JSON-specific operators like json_contains, json_has_key, etc."""
        try:
            if operator == 'json_contains':
                if not isinstance(value, (dict, list)):
                    raise ValidationError("json_contains value must be dict or list")
                return query.filter(json_accessor.contains(value))
            
            elif operator == 'json_contained_by':
                if not isinstance(value, (dict, list)):
                    raise ValidationError("json_contained_by value must be dict or list")
                return query.filter(json_accessor.contained_by(value))
            
            elif operator == 'json_has_key':
                if not isinstance(value, str) or not self._is_safe_json_key(value):
                    raise SecurityError(f"Invalid JSON key for has_key: {value}")
                return query.filter(json_accessor.has_key(value))
            
            elif operator == 'json_has_any_key':
                if not isinstance(value, (list, tuple)):
                    raise ValidationError("json_has_any_key value must be list")
                for key in value:
                    if not isinstance(key, str) or not self._is_safe_json_key(key):
                        raise SecurityError(f"Invalid JSON key: {key}")
                return query.filter(json_accessor.has_any(value))
            
            elif operator == 'json_has_all_keys':
                if not isinstance(value, (list, tuple)):
                    raise ValidationError("json_has_all_keys value must be list")
                for key in value:
                    if not isinstance(key, str) or not self._is_safe_json_key(key):
                        raise SecurityError(f"Invalid JSON key: {key}")
                return query.filter(json_accessor.has_all(value))
            
            else:
                raise SecurityError(f"Unsupported JSON operator: {operator}")
        
        except Exception as e:
            if isinstance(e, (SecurityError, ValidationError)):
                raise
            else:
                raise ValidationError(f"JSON operation failed: {e}")

    def _cast_json_value(self, json_accessor, value):
        """
        Cast JSON accessor to appropriate type based on the value being compared.
        This ensures proper type comparison for JSON fields.
        """        
        # Handle boolean FIRST (before int check since bool is subclass of int)
        if isinstance(value, bool):
            return cast(json_accessor.astext, Boolean) if hasattr(json_accessor, 'astext') else cast(json_accessor, Boolean) #?
        elif isinstance(value, int):
            return cast(json_accessor.astext, Integer) if hasattr(json_accessor, 'astext') else cast(json_accessor, Integer) #?
        elif isinstance(value, float):
            return cast(json_accessor.astext, Float) if hasattr(json_accessor, 'astext') else cast(json_accessor, Float) #?
        elif isinstance(value, str):
            return json_accessor.astext if hasattr(json_accessor, 'astext') else json_accessor #?
        elif isinstance(value, (list, tuple)) and value:
            # For IN operations, cast based on first element type
            first_item = value[0]
            if isinstance(first_item, bool):
                return cast(json_accessor.astext, Boolean) if hasattr(json_accessor, 'astext') else cast(json_accessor, Boolean) #?
            elif isinstance(first_item, int):
                return cast(json_accessor.astext, Integer) if hasattr(json_accessor, 'astext') else cast(json_accessor, Integer) #?
            elif isinstance(first_item, float):
                return cast(json_accessor.astext, Float) if hasattr(json_accessor, 'astext') else cast(json_accessor, Float) #?
            else:
                return json_accessor.astext if hasattr(json_accessor, 'astext') else json_accessor #?
        else:
            # For None, complex types, or empty lists - compare as text
            return json_accessor.astext if hasattr(json_accessor, 'astext') else json_accessor #?

    def _is_json_field(self, field) -> bool:
        """Check if a SQLAlchemy field is a JSON type."""
        json_types = (JSON,)
        
        # Add PostgreSQL JSONB if available
        try:
            from sqlalchemy.dialects.postgresql import JSONB
            json_types = (JSON, JSONB)
        except ImportError:
            pass
        
        return isinstance(field.type, json_types)

    def _is_safe_json_key(self, key: str) -> bool:
        """
        Validate that JSON key is safe to use.
        Security: Prevents injection through JSON keys.
        """
        if not isinstance(key, str):
            return False
        
        # Allow alphanumeric, underscore, hyphen
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', key)) and len(key) <= 100
    
    # GENERIC CRUD OPERATIONS
    def create_record(self, record) -> int:
        """Create a new record with validation."""
        model_class = type(record)
        if model_class == Experiment:
            raise SecurityError(f"Create record access denied for type \"{model_class.__name__}\": use \"create_experiment\" instead")
        try:
            with self.get_session() as session:
                session.add(record)
                session.flush()  # Get ID without committing
                record_id = record.id
                
                logger.info(f"Created {model_class.__name__} record with ID: {record_id}")
                return record_id
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error creating {model_class.__name__}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error creating {model_class.__name__}: {e}")
            raise
    
    def read_record(self, model_class: Type[TOrm], record_id: int, as_dict: bool = False) -> Optional[Dict | TOrm]:
        """Read a single record by ID."""
        try:
            with self.get_session() as session:
                record = session.query(model_class).get(record_id) #filter(model_class.id == record_id).first()
                if record:
                    return record.to_dict() if as_dict else record
                else:
                    return None
        except SQLAlchemyError as e:
            logger.error(f"Error reading {model_class.__name__} ID {record_id}: {e}")
            raise
    
    def read_record_where(
            self, model_class: Type[TOrm], conditions: Dict, as_dict: bool = True
    ) -> List[Dict]:
        try:
            with self.get_session() as session:
                query = session.query(model_class).options(undefer('*'))
                query = self._build_query_conditions(query, model_class, conditions)
                record = query.first()
                #record = records[0]
                #if hasattr(record, "parameters"):
                #    print(record.parameters, type(record.parameters), sep='\n')
                if record is not None:
                    return record.to_dict() if as_dict else record
                else:
                    return None
        except (SecurityError, ValidationError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error querying {model_class.__name__}: {e}")
            raise
    
    def read_records(self, model_class: Type[TOrm], limit: int = 1000, offset: int = 0, as_dict: bool = True) -> List[Dict | TOrm]:
        """
        Read all records with pagination.
        Security: Enforces reasonable limits to prevent resource exhaustion.
        """
        if limit > 10000:  # Security: Prevent large queries
            raise SecurityError("Limit too large (max 10000)")
        
        try:
            with self.get_session() as session:
                records = session.query(model_class).offset(offset).limit(limit).all()
                return [record.to_dict() for record in records] if as_dict else records
        except SQLAlchemyError as e:
            logger.error(f"Error reading {model_class.__name__} records: {e}")
            raise
    
    def read_records_where(
            self, model_class: Type[TOrm], conditions: Dict, 
            limit: int = 1000, offset: int = 0, as_dict: bool = True
    ) -> List[Dict]:
        """Read records matching conditions with security validation."""
        if limit > 10000:  # Security: Prevent large queries
            raise SecurityError("Limit too large (max 10000)")
        try:
            with self.get_session() as session:
                query = session.query(model_class).options(undefer('*'))
                query = self._build_query_conditions(query, model_class, conditions)
                records = query.offset(offset).limit(limit).all()
                #record = records[0]
                #if hasattr(record, "parameters"):
                #    print(record.parameters, type(record.parameters), sep='\n')
                return [record.to_dict() for record in records] if as_dict else records
        except (SecurityError, ValidationError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error querying {model_class.__name__}: {e}")
            raise
    
    def update_record(self, model_class: Type[TOrm], record_id: TOrm, data: Dict) -> Optional[dict]:
        """Update a record with validation."""
        try:
            with self.get_session() as session:
                # Find the existing record by primary key (assume 'id' is the PK)
                existing = session.query(model_class).get(record_id)
                if existing:
                    # Update all columns from new_record to existing
                    for attr, value in data.items():
                        if attr in {'tags', 'metadata'}:
                            print("Cannot set \"tags\" and \"metadata\" field with \"update_record\", ignoring them ...")
                        else:
                            setattr(existing, attr, value)
                    return existing.to_dict()
                return None
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {model_class.__name__}.{record_id}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {model_class.__name__}.{record_id}: {e}")
            raise
    
    def update_records(self, model_class: Type[TOrm], record_ids: List[int], data: List[Dict]) -> Tuple[List[dict], List[int]]:
        """Update a list of records."""
        try:
            with self.get_session() as session:
                updated_records, non_existing_records = [], []
                for record_id, item in zip(record_ids, data):
                    updated = self.update_record(model_class, record_id, item)
                    if updated is not None:
                        updated_records.append(updated)
                    else:
                        non_existing_records.append(record_id)
                return updated_records, non_existing_records
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {record_ids}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {record_ids}: {e}")
            raise
    
    def delete_record(self, model_class: Type[TOrm], record_id: int) -> Optional[dict]:
        """Delete a record."""
        try:
            with self.get_session() as session:
                current_record = session.query(model_class).filter(model_class.id == record_id)
                if (current_record is not None) and (current_record.delete()):
                    logger.info(f"Deleted {model_class.__name__} ID {record_id}")
                    return current_record.to_dict()
                else:
                    return None
        except IntegrityError as e:
            logger.error(f"Cannot delete {model_class.__name__} ID {record_id}: {e}")
            raise ValidationError(f"Cannot delete record: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting {model_class.__name__}: {e}")
            raise
    
    def delete_records(self, model_class: Type[TOrm], record_ids: List[int]) -> Tuple[List[dict], List[int]]:
        """Delete a list of records."""
        try:
            with self.get_session() as session:
                deleted_records, non_existing_records = [], []
                for record_id in record_ids:
                    deleted = self.delete_record(model_class, record_id)
                    if deleted is not None:
                        deleted_records.append(deleted)
                    else:
                        non_existing_records.append(record_id)
                return deleted_records, non_existing_records
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {record_ids}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {record_ids}: {e}")
            raise
    
    # Tags and Metadata Updating and Removing (cannot be updated or removed with update_record)
    def add_or_update_tags(self, model_class: Type[TOrm], record_id: int, tags: dict[str, str]) -> Optional[dict]:
        try:
            with self.get_session() as session:
                # Find the existing record by primary key (assume 'id' is the PK)
                existing = session.query(model_class).get(record_id) # ?
                if existing:
                    updated_tags: dict = existing.tags.copy()
                    for tag_name, tag_value in tags.items():
                        updated_tags[tag_name] = tag_value
                    existing.tags = updated_tags
                    flag_modified(existing, "tags")
                    #existing_tags.update(tags)
                    return existing.to_dict()
                return None
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {model_class.__name__}.{record_id}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {model_class.__name__}.{record_id}: {e}")
            raise
    
    def remove_tags(self, model_class: Type[TOrm], record_id: int, tag_names: list[str]):
        try:
            with self.get_session() as session:
                # Find the existing record by primary key (assume 'id' is the PK)
                existing = session.query(model_class).get(record_id) # ?
                if existing:
                    updated_tags: dict = existing.tags.copy()
                    for tag in tag_names:
                        updated_tags.pop(tag)
                    existing.tags = updated_tags
                    flag_modified(existing, 'tags')
                    return existing.to_dict()
                return None
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {model_class.__name__}.{record_id}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {model_class.__name__}.{record_id}: {e}")
            raise

    def add_or_update_metadata(self, model_class: Type[TOrm], record_id: int, metadata: dict[str, Any]):
        try:
            with self.get_session() as session:
                # Find the existing record by primary key (assume 'id' is the PK)
                existing = session.query(model_class).get(record_id) # ?
                if existing:
                    updated_metadata: dict = existing.metadata.copy()
                    updated_metadata.update(metadata)
                    existing.metadata = updated_metadata
                    flag_modified(existing, 'metadata')
                    return existing.to_dict()
                return None
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {model_class.__name__}.{record_id}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {model_class.__name__}.{record_id}: {e}")
            raise
    
    def remove_metadata(self, model_class: Type[TOrm], record_id: int, metadata_names: list[str]):
        try:
            with self.get_session() as session:
                # Find the existing record by primary key (assume 'id' is the PK)
                existing = session.query(model_class).get(record_id) # ?
                if existing:
                    updated_metadata: dict = existing.metadata.copy()
                    for metadata in metadata_names:
                        updated_metadata.pop(metadata)
                    existing.metadata = updated_metadata
                    flag_modified(existing, 'metadata')
                    return existing.to_dict()
                return None
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {model_class.__name__}.{record_id}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {model_class.__name__}.{record_id}: {e}")
            raise

    def count_records(self, model_class: Type[TOrm], conditions: Optional[Dict] = None) -> int:
        """Count records matching conditions."""
        try:
            with self.get_session() as session:
                query = session.query(func.count(model_class.id))
                if conditions:
                    query = self._build_query_conditions(query, model_class, conditions)
                return query.scalar()
        except (SecurityError, ValidationError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error counting {model_class.__name__}: {e}")
            raise
    
    def create_experiment(self, record: Experiment, max_attempts: int = 10) -> Tuple[int, str]:
        """
        Create an experiment with automatic unique name generation.
        Returns:
            Tuple of (experiment_id, final_experiment_name)
        """
        # Validate the provided name
        record.status = 'invalid' # Impose by definition
        name = record.name or 'Experiment'
        if not validate_experiment_name(name):
            raise ValidationError(f"Invalid experiment name: {name}")
        
        original_name = name
        final_name = generate_unique_experiment_name(name)

        with self.get_session() as session:
            for attempt in range(max_attempts):
                record.status = 'init'
                record.name = final_name
                record.start_time = datetime.now()
                try:
                    session.add(record)
                    session.flush()  # Get ID without committing
                    experiment_id = record.id
                    logger.info(f"Created experiment with unique name: {final_name}")
                    return experiment_id, final_name
                except (ValidationError, SecurityError):
                    raise
                except IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e) or "name" in str(e).lower():
                        # Generate a new unique name
                        final_name = generate_unique_experiment_name(original_name)
                        logger.warning(f"Name collision for '{name}', trying '{final_name}'")
                        continue
                    else:
                        # Other integrity error, re-raise
                        raise ValidationError(f"Data integrity violation: {e}")
                except SQLAlchemyError as e:
                    logger.error(f"Database error creating {Experiment.__name__}: {e}")
                    raise
            # If we have exhausted all attempts
            # Restore original name and set status = "invalid"
        record.name = name
        record.status = "invalid"
        raise ValidationError(f"Could not generate unique experiment name after {max_attempts} attempts")
    
    # ADVANCED QUERY METHODS
    def read_experiments_with_details(self, conditions: Optional[Dict] = None,
                                    limit: int = 1000, offset: int = 0) -> List[Dict]:
        """
        Read experiments with all related configuration details.
        Uses eager loading for optimal performance.
        """
        if limit > 10000:
            raise SecurityError("Limit too large (max 10000)")
        
        try:
            with self.get_session() as session:
                query = session.query(Experiment).options(
                    joinedload(Experiment.general),
                    joinedload(Experiment.scenario),
                    joinedload(Experiment.architecture),
                    joinedload(Experiment.loss),
                    joinedload(Experiment.optimizer),
                    joinedload(Experiment.scheduler),
                    joinedload(Experiment.strategy),
                    joinedload(Experiment.active_learning)
                )
                
                if conditions:
                    query = self._build_query_conditions(query, Experiment, conditions)
                
                experiments = query.offset(offset).limit(limit).all()
                return [exp.to_detailed_dict() for exp in experiments]
                
        except (SecurityError, ValidationError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error reading detailed experiments: {e}")
            raise
    
    def search_experiments_advanced(self, general_conditions: Optional[Dict] = None,
                                  scenario_conditions: Optional[Dict] = None,
                                  architecture_conditions: Optional[Dict] = None,
                                  experiment_conditions: Optional[Dict] = None,
                                  limit: int = 1000, offset: int = 0) -> List[Dict]:
        """
        Advanced search across related tables with security validation.
        """
        if limit > 10000:
            raise SecurityError("Limit too large (max 10000)")
        
        try:
            with self.get_session() as session:
                query = session.query(Experiment).options(
                    joinedload(Experiment.general),
                    joinedload(Experiment.scenario),
                    joinedload(Experiment.architecture),
                    joinedload(Experiment.loss),
                    joinedload(Experiment.optimizer),
                    joinedload(Experiment.scheduler),
                    joinedload(Experiment.strategy)
                )
                
                # Join with related tables if conditions are provided
                if general_conditions:
                    query = query.join(General)
                    query = self._build_query_conditions(query, General, general_conditions)
                
                if scenario_conditions:
                    query = query.join(Scenario)
                    query = self._build_query_conditions(query, Scenario, scenario_conditions)
                
                if architecture_conditions:
                    query = query.join(Architecture)
                    query = self._build_query_conditions(query, Architecture, architecture_conditions)
                
                if experiment_conditions:
                    query = self._build_query_conditions(query, Experiment, experiment_conditions)
                
                experiments = query.offset(offset).limit(limit).all()
                return [exp.to_detailed_dict() for exp in experiments]
                
        except (SecurityError, ValidationError):
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error in advanced experiment search: {e}")
            raise
    
    def add_or_update_logs(self, record_id: int, logs: dict[str, str]) -> Optional[dict]:
        try:
            with self.get_session() as session:
                # Find the existing record by primary key (assume 'id' is the PK)
                existing = session.query(Experiment).get(record_id) # ?
                if existing:
                    updated_logs: dict = existing.logs.copy()
                    for log_name, log_value in logs.items():
                        updated_logs[log_name] = log_value
                    existing.logs = updated_logs
                    flag_modified(existing, "logs")
                    #existing_tags.update(tags)
                    return existing.to_dict()
                return None
        except (ValidationError, SecurityError):
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error updating {Experiment.__name__}.{record_id}: {e}")
            raise ValidationError(f"Data integrity violation: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {Experiment.__name__}.{record_id}: {e}")
            raise
    
    # UTILITY AND ANALYTICS METHODS
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive experiment statistics."""
        try:
            with self.get_session() as session:
                stats = {}
                
                # Basic counts
                stats['total_experiments'] = session.query(func.count(Experiment.id)).scalar()
                stats['total_generals'] = session.query(func.count(General.id)).scalar()
                stats['total_scenarios'] = session.query(func.count(Scenario.id)).scalar()
                stats['total_architectures'] = session.query(func.count(Architecture.id)).scalar()
                
                # Status distribution
                status_counts = session.query(
                    Experiment.status, func.count(Experiment.id)
                ).group_by(Experiment.status).all()
                stats['status_distribution'] = {status: count for status, count in status_counts}
                
                # Recent activity (last 30 days)
                thirty_days_ago = datetime.now() - timedelta(days=30)
                stats['recent_experiments'] = session.query(func.count(Experiment.id)).filter(
                    Experiment.start_time >= thirty_days_ago
                ).scalar()
                
                # Average tasks per experiment
                avg_tasks = session.query(func.avg(Experiment.num_tasks)).scalar()
                stats['average_tasks_per_experiment'] = float(avg_tasks) if avg_tasks else 0
                
                return stats
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting experiment statistics: {e}")
            raise
    
    def cleanup_aborted_experiments(self) -> Tuple[int, List[Tuple[int, str]]]:
        """
        Removes data for all experiments with status == "aborted".

        Returns:
            List of tuples (id, name) to be used for cleaning up directories.
        """
        try:
            with self.get_session() as session:
                aborted_experiments = session.query(Experiment).filter(Experiment.status == "aborted")

                # Count and delete
                count = aborted_experiments.count()
                if count > 0:
                    aborted_experiments.delete(synchronize_session=False)
                    results = [(experiment.id, experiment.name) for experiment in aborted_experiments]
                    return count, results
                else:
                    return count, []
        except SQLAlchemyError as e:
            logger.error(f"Error during cleanup: {e}")
            raise
    
    def set_init_to_pending(self, ids: List[int]) -> Tuple[int, List[Experiment]]:
        """
        Sets the status of all these experiments from "init" to "pending".

        Returns:
            List of successfully updated experiments as objects.
        """
        try:
            with self.get_session() as session:
                # Retrieve all experiments with id in ids and status == "init"
                experiments = session.query(Experiment).filter(
                    Experiment.id.in_(ids),
                    Experiment.status == "init"
                ).all()
                for experiment in experiments:
                    experiment.status = "pending"
                retrieved_ids = [exp.id for exp in experiments]
                # Now write back modified experiments to database, and retrieve updated records
                session.flush()  # Ensure changes are written to the DB
                updated_experiments = session.query(Experiment).filter(
                    Experiment.id.in_(retrieved_ids),
                    Experiment.status == "pending"
                ).all()
                return len(updated_experiments), updated_experiments
        except SQLAlchemyError as e:
            logger.error(f"Error during setting to \"pending\": {e}")
            raise
    
    def set_pending_to_running(self, ids: List[int]):
        """
        Sets the status of all these experiments from "pending" to "running".

        Returns:
            List of successfully updated experiments as objects.
        """
        try:
            with self.get_session() as session:
                # Retrieve all experiments with id in ids and status == "init"
                experiments = session.query(Experiment).filter(
                    Experiment.id.in_(ids),
                    Experiment.status == "pending"
                ).all()
                for experiment in experiments:
                    experiment.status = "running"
                retrieved_ids = [exp.id for exp in experiments]
                # Now write back modified experiments to database, and retrieve updated records
                session.flush()  # Ensure changes are written to the DB
                updated_experiments = session.query(Experiment).filter(
                    Experiment.id.in_(retrieved_ids),
                    Experiment.status == "running"
                ).all()
                return len(updated_experiments), updated_experiments
        except SQLAlchemyError as e:
            logger.error(f"Error during setting to \"running\": {e}")
            raise
    
    def set_any_to_aborted(self, ids: List[int]):
        """
        Sets the status of all these experiments from any status to "aborted".

        Returns:
            List of successfully updated experiments as objects.
        """
        try:
            with self.get_session() as session:
                # Retrieve all experiments with id in ids and status == "init"
                experiments = session.query(Experiment).filter(
                    Experiment.id.in_(ids)
                ).all()
                for experiment in experiments:
                    experiment.status = "aborted"
                retrieved_ids = [exp.id for exp in experiments]
                # Now write back modified experiments to database, and retrieve updated records
                session.flush()  # Ensure changes are written to the DB
                updated_experiments = session.query(Experiment).filter(
                    Experiment.id.in_(retrieved_ids),
                    Experiment.status == "aborted"
                ).all()
                return len(updated_experiments), updated_experiments
        except SQLAlchemyError as e:
            logger.error(f"Error during setting to \"aborted\": {e}")
            raise
    
    def set_running_to_finished(self, ids: List[int], stop_time: Optional[datetime] = None):
        """
        Sets the status of all these experiments from "running" to "finished".
        If stop_time is None, computes stop_time := datetime.now().
        Sets experiment.end_time = stop_time for each experiment.

        Returns:
            List of successfully updated experiments as objects.
        """
        if stop_time is None:
            stop_time = datetime.now()
        try:
            with self.get_session() as session:
                # Retrieve all experiments with id in ids and status == "init"
                experiments = session.query(Experiment).filter(
                    Experiment.id.in_(ids),
                    Experiment.status == "running"
                ).all()
                for experiment in experiments:
                    experiment.status = "finished"
                    experiment.end_time = stop_time
                retrieved_ids = [exp.id for exp in experiments]
                # Now write back modified experiments to database, and retrieve updated records
                session.flush()  # Ensure changes are written to the DB
                updated_experiments = session.query(Experiment).filter(
                    Experiment.id.in_(retrieved_ids),
                    Experiment.status == "finished"
                ).all()
                return len(updated_experiments), updated_experiments
        except SQLAlchemyError as e:
            logger.error(f"Error during setting to \"finished\": {e}")
            raise

    # TODO: Is this method actually useful?
    def cleanup_orphaned_configs(self) -> Dict[str, int]:
        """
        Remove configuration records that are not referenced by any experiment.
        
        Returns:
            Dictionary with count of deleted records per table
        """
        deleted_counts = {}
        
        try:
            with self.get_session() as session:
                # Define tables and their foreign key references in experiments
                cleanup_targets = [
                    (General, Experiment.id_general),
                    (Scenario, Experiment.id_scenario),
                    (Architecture, Experiment.id_architecture),
                    (Loss, Experiment.id_loss),
                    (Optimizer, Experiment.id_optimizer),
                    (Scheduler, Experiment.id_scheduler),
                    (Strategy, Experiment.id_strategy)
                ]
                
                for model_class, foreign_key_field in cleanup_targets:
                    # Find orphaned records
                    subquery = session.query(foreign_key_field).subquery()
                    orphaned = session.query(model_class).filter(
                        ~model_class.id.in_(subquery)
                    )
                    
                    # Count and delete
                    count = orphaned.count()
                    if count > 0:
                        orphaned.delete(synchronize_session=False)
                        deleted_counts[model_class.__name__] = count
                        logger.info(f"Deleted {count} orphaned {model_class.__name__} records")
                
                return deleted_counts
        except SQLAlchemyError as e:
            logger.error(f"Error during cleanup: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database structure and content."""
        try:
            with self.get_session() as session:
                info = {
                    'database_url': str(self.engine.url),
                    'tables': {},
                    'total_records': 0
                }
                
                # Get table information
                for model_class in [
                    General, Scenario, Architecture, Loss, Optimizer, Scheduler, EarlyStopping, Strategy, ActiveLearning, Experiment
                ]:
                    table_name = model_class.__tablename__
                    record_count = session.query(func.count(model_class.id)).scalar()
                    
                    info['tables'][table_name] = {
                        'record_count': record_count,
                        'model_class': model_class.__name__
                    }
                    info['total_records'] += record_count
                return info
        except SQLAlchemyError as e:
            logger.error(f"Error getting database info: {e}")
            raise
    
    def dump_db_to_json(self, out_file: str) -> Optional[Dict]:
        model_classes = [General, Scenario, Architecture, Loss, Optimizer, Scheduler, Strategy, Experiment]
        try:
            with self.get_session() as session:
                data = {model_class.__tablename__: [] for model_class in model_classes}
                for model_class in model_classes:
                    table_name = model_class.__tablename__
                    records = session.query(model_class).all()
                    data[table_name] = [record.to_dict() for record in records]
                with open(out_file, 'w') as fp:
                    json.dump(data, fp, indent=2)
                return data
        except SQLAlchemyError as e:
            logger.error(f"Error getting database info: {e}")
            raise


__all__ = ['SecureMLExperimentDB', 'DEFAULT_DB_FILE', 'DEFAULT_DB_TEST_FILE']
