# Generated with Claude 4 Sonnet, with further modifies by Salvatore Correnti
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, List, Optional, TypeVar, Literal
from typing import Any as TAny
from schema import Schema, And, Any, Or, Use
import uuid
import re


# Create base class for all models
Base: type = declarative_base()

TBase = TypeVar('TBase', bound=Base)

# Security: Define allowed operators to prevent SQL injection
ALLOWED_OPERATORS = {
    'eq': lambda field, value: field == value,
    'ne': lambda field, value: field != value,
    'lt': lambda field, value: field < value,
    'le': lambda field, value: field <= value,
    'gt': lambda field, value: field > value,
    'ge': lambda field, value: field >= value,

    '==': lambda field, value: field == value,
    '!=': lambda field, value: field != value,
    '<': lambda field, value: field < value,
    '<=': lambda field, value: field <= value,
    '>': lambda field, value: field > value,
    '>=': lambda field, value: field >= value,
    
    'like': lambda field, value: field.like(value),
    'ilike': lambda field, value: field.ilike(value),  # Case-insensitive LIKE
    'in': lambda field, value: field.in_(value),
    'not_in': lambda field, value: ~field.in_(value),
    'is_null': lambda field, value: field.is_(None) if value else field.isnot(None),
    'contains': lambda field, value: field.contains(value),  # For JSON fields
    'startswith': lambda field, value: field.like(f"{value}%"),
    'endswith': lambda field, value: field.like(f"%{value}"),
    
    # JSON-specific operators
    'json_contains': lambda field, value: field.contains(value),
    'json_contained_by': lambda field, value: field.contained_by(value),
    'json_has_key': lambda field, key: field.has_key(key),
    'json_has_any_key': lambda field, keys: field.has_any(keys),
    'json_has_all_keys': lambda field, keys: field.has_all(keys)
}

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def generate_unique_experiment_name(base_name: str) -> str:
    """
    Generate a unique experiment name suitable for folder naming.
    
    Args:
        base_name: Base name for the experiment
        
    Returns:
        A unique, filesystem-safe experiment name
    """
    # Sanitize base name for filesystem compatibility
    safe_base = re.sub(r'[^\w\-_.]', '_', base_name)
    safe_base = re.sub(r'_+', '_', safe_base).strip('_')
    
    # Limit length to ensure total name stays reasonable (take first 31 characters or pad them)
    safe_base = safe_base[:31]
    
    # Generate short UUID suffix for uniqueness
    unique_suffix = ''.join(str(uuid.uuid4()).split('-'))  # 32 characters given by uuid.uuid4()
    
    # Combine with timestamp for additional uniqueness and sorting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # 15 characters
    
    # len(safe_base) + 15 + 32 + 2 (traits) = len(safe_base) + 49 characters if len(safe_base) > 0 (52 if safe_base == 'Exp')
    # 15 + 32 + 1 (trait) = 48 if len(safe_base) == 0
    return f"{safe_base}-{timestamp}-{unique_suffix}" if len(safe_base) > 0 else f"{timestamp}-{unique_suffix}"


def validate_experiment_name(name: str) -> bool:
    """
    Validate experiment name for filesystem compatibility.
    
    Args:
        name: The experiment name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or len(name) > 200:
        return False
    
    # Check for invalid characters for cross-platform filesystem compatibility
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    if re.search(invalid_chars, name):
        return False
    
    # Check for reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
        'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
        'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    if name.upper().split('.')[0] in reserved_names:
        return False
    
    return True


# Common utilities
def positive_int(nullable=False):
    if nullable:
        return Or(And(int, lambda x: x > 0), lambda x: x is None)
    else:
        return And(int, lambda x: x > 0)

def geq_int(low: int):
    return And(int, lambda x: x >= low)

def leq_int(high: int):
    return And(int, lambda x: x <= high)

def positive_float():
    return And(float, lambda x: x > 0)

def geq_float(low: int):
    return And(float, lambda x: x >= low)

def leq_float(high: int):
    return And(float, lambda x: x <= high)

def standard_string(
        max_length: int, case: Optional[Literal['upper', 'lower']] = None,
        choices: Optional[list[str]] = None
):
    if case == 'upper':
        if choices:
            return And(str, lambda x: len(x) <= max_length, Use(lambda x: x.upper().strip()), lambda x: x in choices)
        else:
            return And(str, lambda x: len(x) <= max_length, Use(lambda x: x.upper().strip()))
    elif case == 'lower':
        if choices:
            return And(str, lambda x: len(x) <= max_length, Use(lambda x: x.lower().strip()), lambda x: x in choices)
        else:
            return And(str, lambda x: len(x) <= max_length, Use(lambda x: x.lower().strip()))
    elif case is None:
        if choices:
            return And(str, lambda x: len(x) <= max_length, Use(lambda x: x.lower().strip()))
        else:
            return And(str, lambda x: len(x) <= max_length)
    else:
        raise ValueError(f"Unknown case \"{case}\"")

# dict[str, str]
def tags_dict():
    return And(
        dict,
        lambda x: all([isinstance(k, str) for k in x.keys()]),
        lambda x: all([isinstance(v, str) for v in x.values()])
    )

def metadata_dict():
    return And(
        dict,
        lambda x: all([isinstance(k, str) for k in x.keys()]),
        #lambda x: all([any([isinstance(v, t) for t in {int, float, bool, str, list, tuple, set, dict}]) for v in x.values()])
    )

def templated_dict(required_fields: dict[str, type], optional_fields: dict[str, tuple[type, TAny]]):
    def validator(x):
        for field, t in required_fields.items():
            if (field not in x) or (not isinstance(x[field], t)):
                raise
        for field, (t, default) in optional_fields.items():
            if (field in x) and (not isinstance(x[field], t)):
                raise
            elif (field not in x) and (default is not None):
                x[field] = default
        valid_fields = set(required_fields + optional_fields)
        for x_field in x:
            if x_field not in valid_fields:
                raise
        return x
    return And(dict, validator)


__all__ = [
    'Base', 'TBase', 'ALLOWED_OPERATORS', 'SecurityError', 'ValidationError',
    'generate_unique_experiment_name', 'validate_experiment_name',
    'positive_int', 'geq_int', 'leq_int', 'positive_float',
    'geq_float', 'leq_float', 'standard_string',
    'tags_dict', 'metadata_dict', 'templated_dict'
]
