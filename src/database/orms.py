# Generated with Claude 4 Sonnet, with further modifies by Salvatore Correnti
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, UniqueConstraint, Numeric, Float
)
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Optional, TypeVar, Literal
from schema import Schema, And, Any, Or, Use
from schema import Optional as SchemaOpt
import re
import json

from .utils import *


class SchemaORM:
    
    @classmethod
    def from_dict(cls, args: Dict):
        #args = cls._VALIDATION_SCHEMA.validate(args)
        return cls(**args)
    
    @classmethod
    def json_fields(cls) -> List[str]:
        return ['tags', 'other_metadata']


# MODEL DEFINITIONS
class General(Base, SchemaORM):
    
    __tablename__ = 'generals'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'mode': standard_string(32, 'upper'),
        'num_campaigns': positive_int(),
        'train_mb_size': positive_int(),
        'eval_mb_size': positive_int(),
        'train_epochs': positive_int(),
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })

    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(32), nullable=False, index=True) # "CL", "CLAEA" or other modes
    num_campaigns = Column(Integer, nullable=False)
    train_mb_size = Column(Integer, nullable=False)
    eval_mb_size = Column(Integer, nullable=False)
    train_epochs = Column(Integer, nullable=False)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="general") # One-to-Many

    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, mode='{self.mode}', campaigns={self.num_campaigns}, " + \
            f"epochs={self.train_epochs}, mb sizes=({self.train_mb_size}, {self.eval_mb_size}))>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary with security filtering."""
        return {
            'id': self.id,
            'mode': self.mode,
            'num_campaigns': self.num_campaigns,
            'train_mb_size': self.train_mb_size,
            'eval_mb_size': self.eval_mb_size,
            'train_epochs': self.train_epochs,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }    


class Scenario(Base, SchemaORM):
    __tablename__ = 'scenarios'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        SchemaOpt('simulator_type'): standard_string(128, 'lower', ['qualikiz', 'tglf']),
        'pow_type': standard_string(128, 'lower', ['highpow', 'lowpow', 'mixed']),
        'cluster_type': standard_string(128, 'lower', ['tau_based', 'Ip_Pin_based', 'wmhd_based', 'beta_based']),
        'dataset_type': standard_string(128, 'lower', ['not_null', 'complete']),
        'task': standard_string(128, 'lower', ['regression', 'classification']),
        'input_columns': Or(
            standard_string(256, 'lower'), # Either already a string
            And(list, Use(lambda x: '+'.join([y.lower().strip() for y in x]))) # Or a list that gets converted to a string
        ),
        'output_columns': Or(
            standard_string(256, 'lower'), # Either already a string
            And(list, Use(lambda x: '+'.join([y.lower().strip() for y in x]))) # Or a list that gets converted to a string
        ),
        'normalize_inputs': bool,
        'normalize_outputs': bool,
        SchemaOpt('normalization_type'): standard_string(128, 'lower', ['no-normalization', 'first-exp', 'per-exp']),
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulator_type = Column(String(128), nullable=False, index=True, default='qualikiz')
    pow_type = Column(String(128), nullable=False)
    cluster_type = Column(String(128), nullable=False)
    dataset_type = Column(String(128), nullable=False, index=True)
    task = Column(String(128), nullable=False, index=True, default="regression")
    input_columns = Column(Text, nullable=False)  # JSON string of list
    output_columns = Column(Text, nullable=False)  # JSON string of list
    normalize_inputs = Column(Boolean, nullable=False, default=True)
    normalize_outputs = Column(Boolean, nullable=False, default=False)
    normalization_type = Column(String(128), nullable=False, default="first-exp") # "no-normalization", "first-exp", "per-exp"
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="scenario")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        if 'input_columns' in kwargs:
            print("**********" + kwargs['input_columns'])
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, desc='{self.simulator_type}-{self.pow_type}-{self.cluster_type}', " + \
            f"dataset_type='{self.dataset_type}', task='{self.task}', inputs={self.input_columns}, outputs={self.output_columns}, " + \
            f"normalization = '{self.normalization_type} ({self.normalize_inputs}, {self.normalize_outputs})')>"
    
    @property
    def input_columns_list(self) -> List[str]:
        """Get input columns as a list."""
        try:
            return json.loads(self.input_columns) if self.input_columns else []
        except json.JSONDecodeError:
            # Fallback for legacy "+" separated format
            return self.input_columns.split('+') if self.input_columns else []
    
    @property
    def output_columns_list(self) -> List[str]:
        """Get output columns as a list."""
        try:
            return json.loads(self.output_columns) if self.output_columns else []
        except json.JSONDecodeError:
            # Fallback for legacy "+" separated format
            return self.output_columns.split('+') if self.output_columns else []
    
    def set_input_columns(self, columns: List[str]):
        """Set input columns from a list."""
        self.input_columns = json.dumps(columns)
    
    def set_output_columns(self, columns: List[str]):
        """Set output columns from a list."""
        self.output_columns = json.dumps(columns)
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'simulator_type': self.simulator_type,
            'pow_type': self.pow_type,
            'cluster_type': self.cluster_type,
            'dataset_type': self.dataset_type,
            'task': self.task,
            'input_columns': self.input_columns_list,
            'output_columns': self.output_columns_list,
            'normalize_inputs': self.normalize_inputs,
            'normalize_outputs': self.normalize_outputs,
            "normalization_type": self.normalization_type,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }


class Architecture(Base, SchemaORM):
    __tablename__ = 'architectures'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'model_type': standard_string(256),
        'model_folder': standard_string(512),
        SchemaOpt('parameters'): dict,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(256), nullable=False, index=True, default="MLP")
    model_folder = Column(String(512), nullable=True)
    parameters = Column(JSON, nullable=True)  # Native JSON support
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="architecture")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, type='{self.model_type}', parameters={self.parameters})>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_folder': self.model_folder,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class Loss(Base, SchemaORM):
    __tablename__ = 'losses'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'name': standard_string(128, case='lower'),
        SchemaOpt('parameters'): dict,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True)
    parameters = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="loss")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class Optimizer(Base, SchemaORM):
    __tablename__ = 'optimizers'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'name': standard_string(128),
        SchemaOpt('parameters'): dict,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True)
    parameters = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="optimizer")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class Scheduler(Base, SchemaORM):
    __tablename__ = 'schedulers'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'name': standard_string(128),
        SchemaOpt('parameters'): dict,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True)
    parameters = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="scheduler")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class EarlyStopping(Base, SchemaORM):
    __tablename__ = 'earlystoppings'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        SchemaOpt('patience'): positive_int(),
        SchemaOpt('metric'): standard_string(128),
        SchemaOpt('delta'): positive_float(),
        SchemaOpt('type'): standard_string(32, 'lower', ['min', 'max']),
        SchemaOpt('restore_best_weights'): bool,
        SchemaOpt('when_above'): Or(int, float),
        SchemaOpt('when_below'): Or(int, float),
        SchemaOpt('min_epochs'): positive_int(),
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patience = Column(Integer, nullable=False, default=50)
    metric = Column(String(128), nullable=False, default='Loss')
    delta = Column(Float, nullable=False, default=0.1)
    type = Column(String(32), nullable=False, default='min')
    restore_best_weights = Column(Boolean, nullable=False, default=True)
    when_above = Column(Float, nullable=False, default=float('-inf'))
    when_below = Column(Float, nullable=False, default=float('inf'))
    min_epochs = Column(Integer, nullable=False, default=100)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)    
    
    # Relationship
    experiments = relationship("Experiment", back_populates="early_stopping")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, metric={self.metric}, type={self.type}, " + \
            f"patience={self.patience}, delta={self.delta}, min_epochs={self.min_epochs})>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'patience': self.patience,
            'metric': self.metric,
            'delta': self.delta,
            'type': self.type,
            'restore_best_weights': self.restore_best_weights,
            'when_above': self.when_above,
            'when_below': self.when_below,
            'min_epochs': self.min_epochs,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }


class Strategy(Base, SchemaORM):
    __tablename__ = 'strategies'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'name': standard_string(128),
        SchemaOpt('from_scratch'): bool,
        SchemaOpt('parameters'): dict,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True)
    from_scratch = Column(Boolean, nullable=False, default=False)
    parameters = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Relationship
    experiments = relationship("Experiment", back_populates="strategy")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'from_scratch': self.from_scratch,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class ActiveLearning(Base, SchemaORM):
    __tablename__ = 'activelearnings'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        SchemaOpt('framework'): standard_string(128, 'lower', ['bmdal']),
        SchemaOpt('batch_size'): positive_int(),
        SchemaOpt('max_batch_size'): positive_int(),
        SchemaOpt('reload_initial_weights'): bool,
        SchemaOpt('standard_method'): standard_string(
            128, 'lower', ["random_sketch_grad", "bald", "batchbald", "badge", "coreset", "bait", "lcmd_sketch_grad"]
        ),
        SchemaOpt('custom_method'): templated_dict(
            {'selection_method': str, 'initial_selection_method': str, 'base_kernel': str},
            {'kernel_transforms': (list, None), 'sel_with_train': (bool, False), 'sigma': (float, 0.01)}
        ),
        SchemaOpt('full_first_set'): bool,
        SchemaOpt('first_set_size'): int,
        SchemaOpt('downsampling_factor'): float,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    framework = Column(String(128), nullable=False, default='bmdal')
    batch_size = Column(Integer, nullable=False, default=256)
    max_batch_size = Column(Integer, nullable=False, default=1024)
    reload_initial_weights = Column(Boolean, nullable=False, default=False)
    standard_method = Column(
        String(128), nullable=False, default="random_sketch_grad",
    )
    custom_method = Column(JSON, nullable=True)
    full_first_set = Column(Boolean, nullable=False, default=False)
    first_set_size = Column(Integer, nullable=False, default=5120)
    downsampling_factor = Column(Float, nullable=False, default=0.5)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)

    # Relationship
    experiments = relationship("Experiment", back_populates="active_learning")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id})>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "framework": self.framework,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "reload_initial_weights": self.reload_initial_weights,
            "standard_method": self.standard_method,
            "custom_method": self.custom_method,
            "full_first_set": self.full_first_set,
            "first_set_size": self.first_set_size,
            "downsampling_factor": self.downsampling_factor,
            "tags": self.tags,
            "other_metadata": self.other_metadata,
        }

    @classmethod
    def json_fields(cls):
        return ['custom_method'] + SchemaORM.json_fields()


class Experiment(Base, SchemaORM):
    __tablename__ = 'experiments'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'id_general': positive_int(),
        'id_scenario': positive_int(),
        'id_architecture': positive_int(),
        'id_loss': positive_int(),
        'id_optimizer': positive_int(),
        'id_scheduler': positive_int(),
        'id_early_stopping': positive_int(),
        'id_strategy': positive_int(),
        SchemaOpt('id_active_learning'): positive_int(nullable=True),
        SchemaOpt('name'): standard_string(128),
        SchemaOpt('start_time'): datetime,
        SchemaOpt('end_time'): datetime,
        'num_tasks': positive_int(),
        SchemaOpt('status'): standard_string(32, 'lower', ['invalid', 'init', 'pending', 'running', 'aborted', 'finished']),
        SchemaOpt('logs'): dict,
        SchemaOpt('is_test'): bool,
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    id_general = Column(Integer, ForeignKey('generals.id'), nullable=False, index=True)
    id_scenario = Column(Integer, ForeignKey('scenarios.id'), nullable=False, index=True)
    id_architecture = Column(Integer, ForeignKey('architectures.id'), nullable=False, index=True)
    id_loss = Column(Integer, ForeignKey('losses.id'), nullable=False, index=True)
    id_optimizer = Column(Integer, ForeignKey('optimizers.id'), nullable=False, index=True)
    id_scheduler = Column(Integer, ForeignKey('schedulers.id'), nullable=False, index=True)
    id_early_stopping = Column(Integer, ForeignKey('earlystoppings.id'), nullable=False, index=True)
    id_strategy = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)
    id_active_learning = Column(Integer, ForeignKey('activelearnings.id'), nullable=True, index=True)
    name = Column(String(128), nullable=False, unique=True, index=True)  # UNIQUE constraint added
    start_time = Column(DateTime, nullable=False, default=datetime.now(), index=True)
    end_time = Column(DateTime, nullable=True, index=True)
    num_tasks = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default='invalid', index=True) # "invalid", "init", "pending", "running", "aborted", "finished"
    logs = Column(JSON, nullable=True)
    is_test = Column(Boolean, nullable=False, default=False)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)
    
    # Add additional constraint for extra safety
    __table_args__ = (
        UniqueConstraint('name', name='uq_experiment_name'),
    )
    
    # Relationships
    general = relationship("General", back_populates="experiments")
    scenario = relationship("Scenario", back_populates="experiments")
    architecture = relationship("Architecture", back_populates="experiments")
    loss = relationship("Loss", back_populates="experiments")
    optimizer = relationship("Optimizer", back_populates="experiments")
    scheduler = relationship("Scheduler", back_populates="experiments")
    early_stopping = relationship("EarlyStopping", back_populates="experiments")
    strategy = relationship("Strategy", back_populates="experiments")
    active_learning = relationship("ActiveLearning", back_populates="experiments")
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(kwargs)
        super().__init__(**kwargs)
    
    def __repr__(self):
        desc = ', '.join([f"{k}={v}" for k, v in self.to_dict().items()])
        return f"<{type(self).__name__}({desc})>"
    
    @property
    def folder_safe_name(self) -> str:
        """Get a filesystem-safe version of the experiment name."""
        return re.sub(r'[^\w\-_.]', '_', self.name)
        
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'id_general': self.id_general,
            'id_scenario': self.id_scenario,
            'id_architecture': self.id_architecture,
            'id_loss': self.id_loss,
            'id_optimizer': self.id_optimizer,
            'id_scheduler': self.id_scheduler,
            'id_early_stopping': self.id_early_stopping,
            'id_strategy': self.id_strategy,
            'id_active_learning': self.id_active_learning,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'num_tasks': self.num_tasks,
            'status': self.status,
            'logs': self.logs,
            'is_test': self.is_test,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        }
    
    def to_detailed_dict(self) -> Dict:
        """Convert model to dictionary with all related data."""
        result = self.to_dict()
        if self.general:
            result.update({f"general_{k}": v for k, v in self.general.to_dict().items() if k != 'id'})
        if self.scenario:
            result.update({f"scenario_{k}": v for k, v in self.scenario.to_dict().items() if k != 'id'})
        if self.architecture:
            result.update({f"architecture_{k}": v for k, v in self.architecture.to_dict().items() if k != 'id'})
        if self.loss:
            result.update({f"loss_{k}": v for k, v in self.loss.to_dict().items() if k != 'id'})
        if self.optimizer:
            result.update({f"optimizer_{k}": v for k, v in self.optimizer.to_dict().items() if k != 'id'})
        if self.scheduler:
            result.update({f"scheduler_{k}": v for k, v in self.scheduler.to_dict().items() if k != 'id'})
        if self.early_stopping:
            result.update({f"early_stopping_{k}": v for k, v in self.early_stopping.to_dict().items() if k != 'id'})
        if self.strategy:
            result.update({f"strategy_{k}": v for k, v in self.strategy.to_dict().items() if k != 'id'})
        if self.active_learning:
            result.update({f"active_learning_{k}": v for k, v in self.strategy.to_dict().items() if k != 'id'})
        return result

    @classmethod
    def json_fields(cls):
        return ['logs'] + SchemaORM.json_fields()


TOrm = TypeVar(
    'Orm', 
    *[
        SchemaORM, General, Scenario, Architecture, Loss, Optimizer,
        Scheduler, EarlyStopping, Strategy, ActiveLearning, Experiment
    ]
)


__all__ = [
    'General', 'Scenario', 'Architecture', 'Loss', 'Optimizer', 'Scheduler',
    'EarlyStopping', 'Strategy', 'ActiveLearning', 'Experiment', 'TOrm'
]
