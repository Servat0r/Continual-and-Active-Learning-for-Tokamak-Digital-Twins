# Generated with Claude 4 Sonnet, with further modifies by Salvatore Correnti
from abc import abstractmethod
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, UniqueConstraint, Numeric, Float
)
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, TypeVar
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
    
    @classmethod
    @abstractmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        pass


# MODEL DEFINITIONS
class General(Base, SchemaORM):
    
    __tablename__ = 'generals'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        'mode': standard_string(32),
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

    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        match args['mode']:
            case 'cl' | 'CL':
                args['mode'] = 'CL'
            case 'claea' | 'CLAEA' | 'al(cl)' | 'AL(CL)':
                args['mode'] = 'CLAEA'
            case _:
                args['mode'] = args['mode'].upper()
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, mode='{self.mode}', campaigns={self.num_campaigns}, " + \
            f"epochs={self.train_epochs}, mb sizes=({self.train_mb_size}, {self.eval_mb_size}))>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary with security filtering."""
        return self.destandardize_fields({
            'id': self.id,
            'mode': self.mode,
            'num_campaigns': self.num_campaigns,
            'train_mb_size': self.train_mb_size,
            'eval_mb_size': self.eval_mb_size,
            'train_epochs': self.train_epochs,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        })


class Scenario(Base, SchemaORM):
    __tablename__ = 'scenarios'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        SchemaOpt('simulator_type'): standard_string(128, choices=['qualikiz', 'tglf']),
        'pow_type': standard_string(128, choices=['highpow', 'lowpow', 'mixed']),
        'cluster_type': standard_string(128, choices=['tau_based', 'Ip_Pin_based', 'wmhd_based', 'beta_based']),
        'dataset_type': standard_string(128, choices=['not_null', 'complete']),
        'task': standard_string(128, choices=['regression', 'classification']),
        'input_columns': Or(standard_string(256), list), # Either string or list
        'output_columns': Or(standard_string(256), list), # Either string or list
        'normalize_inputs': bool,
        'normalize_outputs': bool,
        SchemaOpt('normalization_type'): standard_string(128, choices=['no-normalization', 'first-exp', 'per-exp']),
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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        for field in ['simulator_type', 'cluster_type', 'dataset_type', 'task']:
            if field in args:
                args[field] = args[field].lower()
        # Standardize simulator type
        if args['simulator_type'] == 'qlk':
            args['simulator_type'] = 'qualikiz'
        # Standardize power type
        if 'pow_type' in args:
            match args['pow_type']:
                case 'hp' | 'high' | 'high_pow':
                    args['pow_type'] = 'highpow'
                case 'lp' | 'low' | 'low_pow':
                    args['pow_type'] = 'lowpow'
                case 'mp' | 'mixed' | 'mixed_pow':
                    args['pow_type'] = 'mixed'
        # Standardize cluster type
        if 'cluster_type' in args:
            match args['cluster_type']:
                case 'ip_pin' | 'ip_pin_based':
                    args['cluster_type'] = 'Ip_Pin_based'
                case 'tau' | 'tau_based':
                    args['cluster_type'] = 'tau_based'
                case 'wmhd' | 'wmhd_based':
                    args['cluster_type'] = 'wmhd_based'
                case 'beta' | 'beta_based':
                    args['cluster_type'] = 'beta_based'
        # Standardize input columns
        if 'input_columns' in args:
            args['input_columns'] = '+'.join([item.lower().strip() for item in args['input_columns']])
        # Standardize output columns
        if 'output_columns' in args:
            args['output_columns'] = '+'.join([item.lower().strip() for item in args['output_columns']])
        # Standardize normalization type
        if 'normalization_type' in args:
            args['normalization_type'] = args['normalization_type'].lower().replace('_', '-')
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        if 'input_columns' in args:
            args['input_columns'] = args['input_columns'].split('+')
        if 'output_columns' in args:
            args['output_columns'] = args['output_columns'].split('+')
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
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
        return self.destandardize_fields({
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
        })


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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        if 'model_type' in args:
            match args['model_type']:
                case 'mlp' | 'MLP':
                    args['model_type'] = 'MLP'
                case 'gaussian_mlp' | 'GaussianMLP' | 'Gaussian_MLP':
                    args['model_type'] = 'GaussianMLP'
                case 'convnet' | 'ConvNet':
                    args['model_type'] = 'ConvNet'
                case 'transformer' | 'Transformer':
                    args['model_type'] = 'Transformer'
                case _:
                    args['model_type'] = args['model_type'].upper()
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, type='{self.model_type}', parameters={self.parameters})>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.destandardize_fields({
            'id': self.id,
            'model_type': self.model_type,
            'model_folder': self.model_folder,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        })

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class Loss(Base, SchemaORM):
    __tablename__ = 'losses'
    
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
    experiments = relationship("Experiment", back_populates="loss")
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        if 'name' in args:
            args['name'] = args['name'].lower()
            match args['name']:
                case 'rmse' | 'rootmse' | 'root_mse':
                    args['name'] = 'rmse'
                case 'bcewithlogits' | 'bce_with_logits':
                    args['name'] = 'bcewithlogits'
                case 'gaussiannll' | 'gaussian_nll':
                    args['name'] = 'gaussiannll'
                case 'msecosinesimilarity' | 'mse_cosine_similarity':
                    args['name'] = 'msecosinesimilarity'
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        if 'name' in args:
            match args['name']:
                case 'rmse' | 'rootmse' | 'root_mse':
                    args['name'] = 'RootMSE'
                case 'bcewithlogits' | 'bce_with_logits':
                    args['name'] = 'BCEWithLogits'
                case 'gaussiannll' | 'gaussian_nll':
                    args['name'] = 'GaussianNLL'
                case 'msecosinesimilarity' | 'mse_cosine_similarity':
                    args['name'] = 'MSECosineSimilarity'
                case _:
                    args['name'] = args['name'].upper()
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.destandardize_fields({
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        })

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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        if 'name' in args:
            args['name'] = args['name'].upper()
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        if 'name' in args:
            match args['name']:
                case 'adam' | 'ADAM':
                    args['name'] = 'Adam'
                case 'adamw' | 'ADAMW':
                    args['name'] = 'AdamW'
                case 'adagrad' | 'ADAGRAD':
                    args['name'] = 'Adagrad'
                case 'rmsprop' | 'RMSPROP':
                    args['name'] = 'RMSprop'
                case _:
                    args['name'] = args['name'].upper()
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.destandardize_fields({
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        })

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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        if 'name' in args:
            args['name'] = args['name'].lower()
            match args['name']:
                case 'steplr' | 'step_lr':
                    args['name'] = 'steplr'
                case 'reducelronplateau' | 'reduce_lr_on_plateau':
                    args['name'] = 'reducelronplateau'
                case 'cosineannealinglr' | 'cosine_annealing_lr':
                    args['name'] = 'cosineannealinglr'
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        if 'name' in args:
            match args['name']:
                case 'steplr':
                    args['name'] = 'StepLR'
                case 'reducelronplateau':
                    args['name'] = 'ReduceLROnPlateau'
                case 'cosineannealinglr':
                    args['name'] = 'CosineAnnealingLR'
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.standardize_fields({
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        })

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
        SchemaOpt('type'): standard_string(32, choices=['min', 'max']),
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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, metric={self.metric}, type={self.type}, " + \
            f"patience={self.patience}, delta={self.delta}, min_epochs={self.min_epochs})>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.destandardize_fields({
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
        })


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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.destandardize_fields({
            'id': self.id,
            'name': self.name,
            'from_scratch': self.from_scratch,
            'parameters': self.parameters,
            'tags': self.tags,
            'other_metadata': self.other_metadata
        })

    @classmethod
    def json_fields(cls):
        return ['parameters'] + SchemaORM.json_fields()


class ActiveLearning(Base, SchemaORM):
    __tablename__ = 'activelearnings'
    
    _VALIDATION_SCHEMA = Schema({
        SchemaOpt('id'): positive_int(),
        SchemaOpt('framework'): standard_string(128, choices=['bmdal']),
        SchemaOpt('batch_size'): positive_int(),
        SchemaOpt('max_batch_size'): positive_int(),
        SchemaOpt('reload_initial_weights'): bool,
        SchemaOpt('standard_method'): And(standard_string(
            128, choices=["random", "random_sketch_grad", "bald", "batchbald", "badge", "coreset", "bait", "lcmd", "lcmd_sketch_grad"]
        ), Use(lambda x: x[-12] if x.endswith('_sketch_grad') else x)),
        SchemaOpt('selection_method'): standard_string(128, 'lower'),
        SchemaOpt('initial_selection_method'): standard_string(128, 'lower'),
        SchemaOpt('base_kernel'): standard_string(128, 'lower'),
        SchemaOpt('kernel_transforms'): Use(lambda x: str(x).lower().strip()),
        SchemaOpt('sel_with_train'): bool,
        SchemaOpt('sigma'): float,
        SchemaOpt('full_first_set'): bool,
        SchemaOpt('first_set_size'): int,
        SchemaOpt('downsampling_factor'): Use(lambda x: float(x)),
        SchemaOpt('tags'): tags_dict(),
        SchemaOpt('other_metadata'): metadata_dict()
    })
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    framework = Column(String(128), nullable=False, default='bmdal')
    batch_size = Column(Integer, nullable=False, default=256)
    max_batch_size = Column(Integer, nullable=False, default=1024)
    reload_initial_weights = Column(Boolean, nullable=False, default=False)
    standard_method = Column(String(128), nullable=True) # TODO: This or "non-standard"?
    selection_method = Column(String(128), nullable=True)
    initial_selection_method = Column(String(128), nullable=True)
    base_kernel = Column(String(128), nullable=True)
    kernel_transforms = Column(String(256), nullable=True)
    sel_with_train = Column(Boolean, nullable=True, default=True)
    sigma = Column(Float, nullable=True, default=0.01)
    full_first_set = Column(Boolean, nullable=False, default=False)
    first_set_size = Column(Integer, nullable=False, default=5120)
    downsampling_factor = Column(Float, nullable=False, default=0.5)
    tags = Column(JSON, nullable=False, default={})
    other_metadata = Column(JSON, nullable=True)

    # Relationship
    experiments = relationship("Experiment", back_populates="active_learning")
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        # Framework
        if 'framework' in args:
            args['framework'] = args['framework'].lower()
        # Standard Method
        if 'standard_method' in args:
            args['standard_method'] = args['standard_method'].lower()
            match args['standard_method']:
                case 'coreset' | 'core_set' | 'core-set':
                    args['standard_method'] = 'coreset'
                case 'batch_bald':
                    args['standard_method'] = 'batchbald'
                case 'lcmd_sketch_grad':
                    args['standard_method'] = 'lcmd'
                case 'random_sketch_grad' | 'random_sketch_ll':
                    args['standard_method'] = 'random'
        # Selection Method and Initial Selection Method
        for field in ['selection_method', 'initial_selection_method']:
            if field in args:
                args[field] = args[field].lower()
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<{type(self).__name__}(id={self.id})>"
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return self.destandardize_fields({
            "id": self.id,
            "framework": self.framework,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "reload_initial_weights": self.reload_initial_weights,
            "standard_method": self.standard_method,
            "selection_method": self.selection_method,
            "initial_selection_method": self.initial_selection_method,
            "base_kernel": self.base_kernel,
            "kernel_transforms": self.kernel_transforms,
            "sel_with_train": self.sel_with_train,
            "sigma": self.sigma,
            "full_first_set": self.full_first_set,
            "first_set_size": self.first_set_size,
            "downsampling_factor": self.downsampling_factor,
            "tags": self.tags,
            "other_metadata": self.other_metadata,
        })


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
        SchemaOpt('status'): standard_string(32, choices=['invalid', 'init', 'pending', 'running', 'aborted', 'finished']),
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
    
    @classmethod
    def standardize_fields(cls, args: Dict) -> Dict:
        return args
    
    @classmethod
    def destandardize_fields(cls, args: Dict) -> Dict:
        return args
    
    def __init__(self, **kwargs):
        kwargs = self._VALIDATION_SCHEMA.validate(self.standardize_fields(kwargs))
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
        return self.destandardize_fields({
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
        })
    
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
