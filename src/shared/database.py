"""
Database models and connection management for federated learning.
Uses SQLAlchemy for ORM and database operations.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

Base = declarative_base()


class TrainingRoundModel(Base):
    """Database model for training rounds."""
    
    __tablename__ = 'training_rounds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, nullable=False, unique=True, index=True)
    global_model_path = Column(String(255))
    participating_clients = Column(Integer, default=0)
    successful_clients = Column(Integer, default=0)
    failed_clients = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    avg_training_loss = Column(Float, default=0.0)
    global_accuracy = Column(Float, default=0.0)
    convergence_score = Column(Float, default=0.0)
    aggregation_time = Column(Float, default=0.0)
    round_duration = Column(Float)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(50), default='waiting')
    config_json = Column(Text)  # JSON string of round configuration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'round_number': self.round_number,
            'global_model_path': self.global_model_path,
            'participating_clients': self.participating_clients,
            'successful_clients': self.successful_clients,
            'failed_clients': self.failed_clients,
            'total_samples': self.total_samples,
            'avg_training_loss': self.avg_training_loss,
            'global_accuracy': self.global_accuracy,
            'convergence_score': self.convergence_score,
            'aggregation_time': self.aggregation_time,
            'round_duration': self.round_duration,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'config': json.loads(self.config_json) if self.config_json else None
        }


class ClientUpdateModel(Base):
    """Database model for client updates."""
    
    __tablename__ = 'client_updates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False, index=True)
    model_update_path = Column(String(255))
    num_samples = Column(Integer, nullable=False)
    training_loss = Column(Float, nullable=False)
    training_accuracy = Column(Float, default=0.0)
    privacy_budget_used = Column(Float, default=0.0)
    compression_ratio = Column(Float, default=1.0)
    training_time = Column(Float, default=0.0)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    status = Column(String(50), default='received')
    metadata_json = Column(Text)  # JSON string of additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'client_id': self.client_id,
            'round_number': self.round_number,
            'model_update_path': self.model_update_path,
            'num_samples': self.num_samples,
            'training_loss': self.training_loss,
            'training_accuracy': self.training_accuracy,
            'privacy_budget_used': self.privacy_budget_used,
            'compression_ratio': self.compression_ratio,
            'training_time': self.training_time,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'status': self.status,
            'metadata': json.loads(self.metadata_json) if self.metadata_json else None
        }


class ClientModel(Base):
    """Database model for client information."""
    
    __tablename__ = 'clients'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(100), nullable=False, unique=True, index=True)
    compute_power = Column(String(20), default='medium')
    network_bandwidth = Column(Integer, default=10)
    available_samples = Column(Integer, default=0)
    supported_models = Column(Text)  # JSON array
    privacy_epsilon = Column(Float, default=1.0)
    privacy_delta = Column(Float, default=1e-5)
    max_grad_norm = Column(Float, default=1.0)
    noise_multiplier = Column(Float, default=1.0)
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    total_rounds = Column(Integer, default=0)
    successful_rounds = Column(Integer, default=0)
    failed_rounds = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    reliability_score = Column(Float, default=1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'client_id': self.client_id,
            'compute_power': self.compute_power,
            'network_bandwidth': self.network_bandwidth,
            'available_samples': self.available_samples,
            'supported_models': json.loads(self.supported_models) if self.supported_models else [],
            'privacy_config': {
                'epsilon': self.privacy_epsilon,
                'delta': self.privacy_delta,
                'max_grad_norm': self.max_grad_norm,
                'noise_multiplier': self.noise_multiplier
            },
            'registered_at': self.registered_at.isoformat() if self.registered_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'is_active': self.is_active,
            'statistics': {
                'total_rounds': self.total_rounds,
                'successful_rounds': self.successful_rounds,
                'failed_rounds': self.failed_rounds,
                'avg_response_time': self.avg_response_time,
                'reliability_score': self.reliability_score
            }
        }


class ClientFailureModel(Base):
    """Database model for client failures."""
    
    __tablename__ = 'client_failures'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False)
    failure_type = Column(String(50), nullable=False)
    failure_details = Column(Text)
    context_json = Column(Text)  # JSON string of failure context
    occurred_at = Column(DateTime, default=datetime.utcnow)
    recovery_attempts = Column(Integer, default=0)
    recovered = Column(Boolean, default=False)
    recovery_time = Column(DateTime)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'client_id': self.client_id,
            'round_number': self.round_number,
            'failure_type': self.failure_type,
            'failure_details': self.failure_details,
            'context': json.loads(self.context_json) if self.context_json else None,
            'occurred_at': self.occurred_at.isoformat() if self.occurred_at else None,
            'recovery_attempts': self.recovery_attempts,
            'recovered': self.recovered,
            'recovery_time': self.recovery_time.isoformat() if self.recovery_time else None
        }


class GlobalModelModel(Base):
    """Database model for global models."""
    
    __tablename__ = 'global_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, nullable=False, unique=True, index=True)
    model_path = Column(String(255), nullable=False)
    model_type = Column(String(50), default='simple_cnn')
    parameter_count = Column(Integer, default=0)
    model_size_bytes = Column(Integer, default=0)
    accuracy_metrics_json = Column(Text)  # JSON string of accuracy metrics
    convergence_score = Column(Float, default=0.0)
    participating_clients_json = Column(Text)  # JSON array of client IDs
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'round_number': self.round_number,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'parameter_count': self.parameter_count,
            'model_size_bytes': self.model_size_bytes,
            'accuracy_metrics': json.loads(self.accuracy_metrics_json) if self.accuracy_metrics_json else {},
            'convergence_score': self.convergence_score,
            'participating_clients': json.loads(self.participating_clients_json) if self.participating_clients_json else [],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SystemMetricsModel(Base):
    """Database model for system metrics snapshots."""
    
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_rounds = Column(Integer, default=0)
    active_clients = Column(Integer, default=0)
    total_clients_ever = Column(Integer, default=0)
    avg_clients_per_round = Column(Float, default=0.0)
    total_samples_processed = Column(Integer, default=0)
    current_global_accuracy = Column(Float, default=0.0)
    best_global_accuracy = Column(Float, default=0.0)
    avg_round_duration = Column(Float, default=0.0)
    system_uptime = Column(Float, default=0.0)
    metrics_json = Column(Text)  # Additional metrics as JSON
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'total_rounds': self.total_rounds,
            'active_clients': self.active_clients,
            'total_clients_ever': self.total_clients_ever,
            'avg_clients_per_round': self.avg_clients_per_round,
            'total_samples_processed': self.total_samples_processed,
            'current_global_accuracy': self.current_global_accuracy,
            'best_global_accuracy': self.best_global_accuracy,
            'avg_round_duration': self.avg_round_duration,
            'system_uptime': self.system_uptime,
            'additional_metrics': json.loads(self.metrics_json) if self.metrics_json else {}
        }


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database manager initialized with URL: {database_url}")
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


class DatabaseRepository:
    """Repository pattern for database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize repository.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    # Training round operations
    
    def create_training_round(self, round_data: Dict[str, Any]) -> TrainingRoundModel:
        """Create a new training round record."""
        try:
            with self.db_manager.get_session() as session:
                round_model = TrainingRoundModel(
                    round_number=round_data['round_number'],
                    participating_clients=round_data.get('participating_clients', 0),
                    started_at=round_data.get('started_at', datetime.utcnow()),
                    status=round_data.get('status', 'waiting'),
                    config_json=json.dumps(round_data.get('config', {}))
                )
                
                session.add(round_model)
                session.commit()
                session.refresh(round_model)
                
                logger.debug(f"Created training round {round_data['round_number']}")
                return round_model
                
        except Exception as e:
            logger.error(f"Failed to create training round: {e}")
            raise
    
    def update_training_round(self, round_number: int, updates: Dict[str, Any]) -> Optional[TrainingRoundModel]:
        """Update training round record."""
        try:
            with self.db_manager.get_session() as session:
                round_model = session.query(TrainingRoundModel).filter(
                    TrainingRoundModel.round_number == round_number
                ).first()
                
                if not round_model:
                    return None
                
                for key, value in updates.items():
                    if hasattr(round_model, key):
                        setattr(round_model, key, value)
                
                session.commit()
                session.refresh(round_model)
                
                logger.debug(f"Updated training round {round_number}")
                return round_model
                
        except Exception as e:
            logger.error(f"Failed to update training round: {e}")
            raise
    
    def get_training_round(self, round_number: int) -> Optional[TrainingRoundModel]:
        """Get training round by number."""
        try:
            with self.db_manager.get_session() as session:
                return session.query(TrainingRoundModel).filter(
                    TrainingRoundModel.round_number == round_number
                ).first()
                
        except Exception as e:
            logger.error(f"Failed to get training round: {e}")
            return None
    
    def get_recent_training_rounds(self, limit: int = 10) -> List[TrainingRoundModel]:
        """Get recent training rounds."""
        try:
            with self.db_manager.get_session() as session:
                return session.query(TrainingRoundModel).order_by(
                    TrainingRoundModel.round_number.desc()
                ).limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to get recent training rounds: {e}")
            return []
    
    # Client operations
    
    def create_or_update_client(self, client_data: Dict[str, Any]) -> ClientModel:
        """Create or update client record."""
        try:
            with self.db_manager.get_session() as session:
                client_model = session.query(ClientModel).filter(
                    ClientModel.client_id == client_data['client_id']
                ).first()
                
                if client_model:
                    # Update existing client
                    for key, value in client_data.items():
                        if hasattr(client_model, key):
                            setattr(client_model, key, value)
                    client_model.last_seen = datetime.utcnow()
                else:
                    # Create new client
                    client_model = ClientModel(
                        client_id=client_data['client_id'],
                        compute_power=client_data.get('compute_power', 'medium'),
                        network_bandwidth=client_data.get('network_bandwidth', 10),
                        available_samples=client_data.get('available_samples', 0),
                        supported_models=json.dumps(client_data.get('supported_models', [])),
                        privacy_epsilon=client_data.get('privacy_epsilon', 1.0),
                        privacy_delta=client_data.get('privacy_delta', 1e-5),
                        max_grad_norm=client_data.get('max_grad_norm', 1.0),
                        noise_multiplier=client_data.get('noise_multiplier', 1.0)
                    )
                    session.add(client_model)
                
                session.commit()
                session.refresh(client_model)
                
                logger.debug(f"Created/updated client {client_data['client_id']}")
                return client_model
                
        except Exception as e:
            logger.error(f"Failed to create/update client: {e}")
            raise
    
    def get_client(self, client_id: str) -> Optional[ClientModel]:
        """Get client by ID."""
        try:
            with self.db_manager.get_session() as session:
                return session.query(ClientModel).filter(
                    ClientModel.client_id == client_id
                ).first()
                
        except Exception as e:
            logger.error(f"Failed to get client: {e}")
            return None
    
    def get_all_clients(self) -> List[ClientModel]:
        """Get all clients."""
        try:
            with self.db_manager.get_session() as session:
                return session.query(ClientModel).all()
                
        except Exception as e:
            logger.error(f"Failed to get all clients: {e}")
            return []
    
    # Client update operations
    
    def create_client_update(self, update_data: Dict[str, Any]) -> ClientUpdateModel:
        """Create client update record."""
        try:
            with self.db_manager.get_session() as session:
                update_model = ClientUpdateModel(
                    client_id=update_data['client_id'],
                    round_number=update_data['round_number'],
                    num_samples=update_data['num_samples'],
                    training_loss=update_data['training_loss'],
                    training_accuracy=update_data.get('training_accuracy', 0.0),
                    privacy_budget_used=update_data.get('privacy_budget_used', 0.0),
                    compression_ratio=update_data.get('compression_ratio', 1.0),
                    training_time=update_data.get('training_time', 0.0),
                    model_update_path=update_data.get('model_update_path'),
                    metadata_json=json.dumps(update_data.get('metadata', {}))
                )
                
                session.add(update_model)
                session.commit()
                session.refresh(update_model)
                
                logger.debug(f"Created client update for {update_data['client_id']}")
                return update_model
                
        except Exception as e:
            logger.error(f"Failed to create client update: {e}")
            raise
    
    def get_client_updates_for_round(self, round_number: int) -> List[ClientUpdateModel]:
        """Get all client updates for a round."""
        try:
            with self.db_manager.get_session() as session:
                return session.query(ClientUpdateModel).filter(
                    ClientUpdateModel.round_number == round_number
                ).all()
                
        except Exception as e:
            logger.error(f"Failed to get client updates for round: {e}")
            return []
    
    # Failure operations
    
    def create_client_failure(self, failure_data: Dict[str, Any]) -> ClientFailureModel:
        """Create client failure record."""
        try:
            with self.db_manager.get_session() as session:
                failure_model = ClientFailureModel(
                    client_id=failure_data['client_id'],
                    round_number=failure_data['round_number'],
                    failure_type=failure_data['failure_type'],
                    failure_details=failure_data.get('failure_details'),
                    context_json=json.dumps(failure_data.get('context', {}))
                )
                
                session.add(failure_model)
                session.commit()
                session.refresh(failure_model)
                
                logger.debug(f"Created failure record for {failure_data['client_id']}")
                return failure_model
                
        except Exception as e:
            logger.error(f"Failed to create client failure: {e}")
            raise
    
    def get_client_failures(self, client_id: Optional[str] = None, limit: int = 100) -> List[ClientFailureModel]:
        """Get client failures."""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(ClientFailureModel)
                
                if client_id:
                    query = query.filter(ClientFailureModel.client_id == client_id)
                
                return query.order_by(ClientFailureModel.occurred_at.desc()).limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to get client failures: {e}")
            return []


def create_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Factory function to create database manager.
    
    Args:
        database_url: Database connection URL (uses environment variable if None)
        
    Returns:
        DatabaseManager: Configured database manager
    """
    if not database_url:
        database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://federated:password@localhost:5432/federated_learning'
        )
    
    return DatabaseManager(database_url)


def init_database(database_url: Optional[str] = None):
    """
    Initialize database with tables.
    
    Args:
        database_url: Database connection URL
    """
    db_manager = create_database_manager(database_url)
    db_manager.create_tables()
    
    # Test connection
    if db_manager.test_connection():
        logger.info("Database initialization completed successfully")
    else:
        raise Exception("Database initialization failed - connection test failed")