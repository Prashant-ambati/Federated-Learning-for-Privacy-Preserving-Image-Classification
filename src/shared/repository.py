"""
Repository pattern implementation for federated learning data persistence.
Provides abstraction layer for data access and storage operations.
"""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Type
import logging
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import SQLAlchemyError

from .database import (
    TrainingRound, ClientUpdate, GlobalModel as GlobalModelDB,
    Client, AggregationResult, get_session
)
from .models import (
    ModelUpdate, GlobalModel, ClientCapabilities, 
    TrainingMetrics, RegistrationResponse
)
from .serialization import ModelUpdateSerializer, GlobalModelSerializer

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    def create(self, entity: Any) -> Any:
        """Create a new entity."""
        pass
    
    @abstractmethod
    def get_by_id(self, entity_id: Any) -> Optional[Any]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def update(self, entity: Any) -> Any:
        """Update an entity."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: Any) -> bool:
        """Delete an entity."""
        pass
    
    @abstractmethod
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Any]:
        """List all entities."""
        pass


class TrainingRoundRepository(BaseRepository):
    """Repository for training round data."""
    
    def __init__(self, session_factory=None):
        """Initialize repository."""
        self.session_factory = session_factory or get_session
        self.serializer = GlobalModelSerializer()
    
    def create(self, round_data: Dict[str, Any]) -> TrainingRound:
        """Create a new training round."""
        try:
            with self.session_factory() as session:
                training_round = TrainingRound(
                    round_number=round_data['round_number'],
                    start_time=round_data.get('start_time', datetime.now()),
                    end_time=round_data.get('end_time'),
                    status=round_data.get('status', 'active'),
                    participating_clients=round_data.get('participating_clients', 0),
                    target_clients=round_data.get('target_clients', 0),
                    global_accuracy=round_data.get('global_accuracy'),
                    global_loss=round_data.get('global_loss'),
                    convergence_score=round_data.get('convergence_score'),
                    aggregation_time=round_data.get('aggregation_time'),
                    configuration=json.dumps(round_data.get('configuration', {}))
                )
                
                session.add(training_round)
                session.commit()
                session.refresh(training_round)
                
                logger.info(f"Created training round {training_round.round_number}")
                return training_round
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create training round: {str(e)}")
            raise
    
    def get_by_id(self, round_id: int) -> Optional[TrainingRound]:
        """Get training round by ID."""
        try:
            with self.session_factory() as session:
                return session.query(TrainingRound).filter(
                    TrainingRound.id == round_id
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get training round {round_id}: {str(e)}")
            return None
    
    def get_by_round_number(self, round_number: int) -> Optional[TrainingRound]:
        """Get training round by round number."""
        try:
            with self.session_factory() as session:
                return session.query(TrainingRound).filter(
                    TrainingRound.round_number == round_number
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get training round {round_number}: {str(e)}")
            return None
    
    def update(self, training_round: TrainingRound) -> TrainingRound:
        """Update training round."""
        try:
            with self.session_factory() as session:
                session.merge(training_round)
                session.commit()
                logger.debug(f"Updated training round {training_round.round_number}")
                return training_round
        except SQLAlchemyError as e:
            logger.error(f"Failed to update training round: {str(e)}")
            raise
    
    def delete(self, round_id: int) -> bool:
        """Delete training round."""
        try:
            with self.session_factory() as session:
                training_round = session.query(TrainingRound).filter(
                    TrainingRound.id == round_id
                ).first()
                
                if training_round:
                    session.delete(training_round)
                    session.commit()
                    logger.info(f"Deleted training round {round_id}")
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete training round {round_id}: {str(e)}")
            return False
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[TrainingRound]:
        """List all training rounds."""
        try:
            with self.session_factory() as session:
                query = session.query(TrainingRound).order_by(desc(TrainingRound.round_number))
                
                if limit:
                    query = query.limit(limit).offset(offset)
                
                return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list training rounds: {str(e)}")
            return []
    
    def get_recent_rounds(self, count: int = 10) -> List[TrainingRound]:
        """Get recent training rounds."""
        return self.list_all(limit=count)
    
    def get_rounds_by_status(self, status: str) -> List[TrainingRound]:
        """Get training rounds by status."""
        try:
            with self.session_factory() as session:
                return session.query(TrainingRound).filter(
                    TrainingRound.status == status
                ).order_by(desc(TrainingRound.round_number)).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get rounds by status {status}: {str(e)}")
            return []
    
    def get_round_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get training round statistics."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.session_factory() as session:
                rounds = session.query(TrainingRound).filter(
                    TrainingRound.start_time >= cutoff_date
                ).all()
                
                if not rounds:
                    return {'message': 'No rounds found in specified period'}
                
                total_rounds = len(rounds)
                completed_rounds = len([r for r in rounds if r.status == 'completed'])
                avg_clients = sum(r.participating_clients or 0 for r in rounds) / total_rounds
                avg_accuracy = sum(r.global_accuracy or 0 for r in rounds if r.global_accuracy) / max(1, len([r for r in rounds if r.global_accuracy]))
                
                return {
                    'period_days': days,
                    'total_rounds': total_rounds,
                    'completed_rounds': completed_rounds,
                    'completion_rate': completed_rounds / total_rounds if total_rounds > 0 else 0,
                    'average_participating_clients': avg_clients,
                    'average_global_accuracy': avg_accuracy,
                    'latest_round': max(r.round_number for r in rounds) if rounds else 0
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get round statistics: {str(e)}")
            return {'error': str(e)}


class ClientUpdateRepository(BaseRepository):
    """Repository for client update data."""
    
    def __init__(self, session_factory=None):
        """Initialize repository."""
        self.session_factory = session_factory or get_session
        self.serializer = ModelUpdateSerializer()
    
    def create(self, model_update: ModelUpdate) -> ClientUpdate:
        """Create a new client update."""
        try:
            with self.session_factory() as session:
                # Serialize model weights
                serialized_weights = self.serializer.serialize_model_update(model_update)
                
                client_update = ClientUpdate(
                    client_id=model_update.client_id,
                    round_number=model_update.round_number,
                    model_weights=pickle.dumps(serialized_weights),
                    num_samples=model_update.num_samples,
                    training_loss=model_update.training_loss,
                    training_accuracy=getattr(model_update, 'training_accuracy', None),
                    privacy_budget_used=model_update.privacy_budget_used,
                    compression_ratio=model_update.compression_ratio,
                    upload_time=model_update.timestamp,
                    validation_status='pending'
                )
                
                session.add(client_update)
                session.commit()
                session.refresh(client_update)
                
                logger.debug(f"Created client update for {model_update.client_id}, round {model_update.round_number}")
                return client_update
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create client update: {str(e)}")
            raise
    
    def get_by_id(self, update_id: int) -> Optional[ClientUpdate]:
        """Get client update by ID."""
        try:
            with self.session_factory() as session:
                return session.query(ClientUpdate).filter(
                    ClientUpdate.id == update_id
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get client update {update_id}: {str(e)}")
            return None
    
    def update(self, client_update: ClientUpdate) -> ClientUpdate:
        """Update client update."""
        try:
            with self.session_factory() as session:
                session.merge(client_update)
                session.commit()
                logger.debug(f"Updated client update {client_update.id}")
                return client_update
        except SQLAlchemyError as e:
            logger.error(f"Failed to update client update: {str(e)}")
            raise
    
    def delete(self, update_id: int) -> bool:
        """Delete client update."""
        try:
            with self.session_factory() as session:
                client_update = session.query(ClientUpdate).filter(
                    ClientUpdate.id == update_id
                ).first()
                
                if client_update:
                    session.delete(client_update)
                    session.commit()
                    logger.info(f"Deleted client update {update_id}")
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete client update {update_id}: {str(e)}")
            return False
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[ClientUpdate]:
        """List all client updates."""
        try:
            with self.session_factory() as session:
                query = session.query(ClientUpdate).order_by(desc(ClientUpdate.upload_time))
                
                if limit:
                    query = query.limit(limit).offset(offset)
                
                return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list client updates: {str(e)}")
            return []
    
    def get_updates_by_round(self, round_number: int) -> List[ClientUpdate]:
        """Get all updates for a specific round."""
        try:
            with self.session_factory() as session:
                return session.query(ClientUpdate).filter(
                    ClientUpdate.round_number == round_number
                ).order_by(ClientUpdate.upload_time).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get updates for round {round_number}: {str(e)}")
            return []
    
    def get_updates_by_client(self, client_id: str, limit: int = 10) -> List[ClientUpdate]:
        """Get recent updates from a specific client."""
        try:
            with self.session_factory() as session:
                return session.query(ClientUpdate).filter(
                    ClientUpdate.client_id == client_id
                ).order_by(desc(ClientUpdate.upload_time)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get updates for client {client_id}: {str(e)}")
            return []
    
    def get_model_update(self, update_id: int) -> Optional[ModelUpdate]:
        """Get deserialized model update."""
        try:
            client_update = self.get_by_id(update_id)
            if not client_update:
                return None
            
            # Deserialize model weights
            serialized_weights = pickle.loads(client_update.model_weights)
            model_update = self.serializer.deserialize_model_update(serialized_weights)
            
            return model_update
            
        except Exception as e:
            logger.error(f"Failed to deserialize model update {update_id}: {str(e)}")
            return None
    
    def cleanup_old_updates(self, days_to_keep: int = 30) -> int:
        """Clean up old client updates."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.session_factory() as session:
                deleted_count = session.query(ClientUpdate).filter(
                    ClientUpdate.upload_time < cutoff_date
                ).delete()
                
                session.commit()
                logger.info(f"Cleaned up {deleted_count} old client updates")
                return deleted_count
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old updates: {str(e)}")
            return 0


class GlobalModelRepository(BaseRepository):
    """Repository for global model data."""
    
    def __init__(self, session_factory=None):
        """Initialize repository."""
        self.session_factory = session_factory or get_session
        self.serializer = GlobalModelSerializer()
    
    def create(self, global_model: GlobalModel) -> GlobalModelDB:
        """Create a new global model."""
        try:
            with self.session_factory() as session:
                # Serialize model
                serialized_model = self.serializer.serialize_global_model(global_model)
                
                global_model_db = GlobalModelDB(
                    round_number=global_model.round_number,
                    model_weights=pickle.dumps(serialized_model),
                    accuracy_metrics=json.dumps(global_model.accuracy_metrics),
                    aggregation_info=json.dumps(global_model.aggregation_info),
                    created_at=datetime.now(),
                    model_size=len(pickle.dumps(serialized_model)),
                    version=f"v{global_model.round_number}"
                )
                
                session.add(global_model_db)
                session.commit()
                session.refresh(global_model_db)
                
                logger.info(f"Created global model for round {global_model.round_number}")
                return global_model_db
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create global model: {str(e)}")
            raise
    
    def get_by_id(self, model_id: int) -> Optional[GlobalModelDB]:
        """Get global model by ID."""
        try:
            with self.session_factory() as session:
                return session.query(GlobalModelDB).filter(
                    GlobalModelDB.id == model_id
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get global model {model_id}: {str(e)}")
            return None
    
    def get_by_round(self, round_number: int) -> Optional[GlobalModelDB]:
        """Get global model by round number."""
        try:
            with self.session_factory() as session:
                return session.query(GlobalModelDB).filter(
                    GlobalModelDB.round_number == round_number
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get global model for round {round_number}: {str(e)}")
            return None
    
    def get_latest(self) -> Optional[GlobalModelDB]:
        """Get the latest global model."""
        try:
            with self.session_factory() as session:
                return session.query(GlobalModelDB).order_by(
                    desc(GlobalModelDB.round_number)
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get latest global model: {str(e)}")
            return None
    
    def update(self, global_model_db: GlobalModelDB) -> GlobalModelDB:
        """Update global model."""
        try:
            with self.session_factory() as session:
                session.merge(global_model_db)
                session.commit()
                logger.debug(f"Updated global model {global_model_db.id}")
                return global_model_db
        except SQLAlchemyError as e:
            logger.error(f"Failed to update global model: {str(e)}")
            raise
    
    def delete(self, model_id: int) -> bool:
        """Delete global model."""
        try:
            with self.session_factory() as session:
                global_model = session.query(GlobalModelDB).filter(
                    GlobalModelDB.id == model_id
                ).first()
                
                if global_model:
                    session.delete(global_model)
                    session.commit()
                    logger.info(f"Deleted global model {model_id}")
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete global model {model_id}: {str(e)}")
            return False
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[GlobalModelDB]:
        """List all global models."""
        try:
            with self.session_factory() as session:
                query = session.query(GlobalModelDB).order_by(desc(GlobalModelDB.round_number))
                
                if limit:
                    query = query.limit(limit).offset(offset)
                
                return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list global models: {str(e)}")
            return []
    
    def get_global_model(self, model_id: int) -> Optional[GlobalModel]:
        """Get deserialized global model."""
        try:
            global_model_db = self.get_by_id(model_id)
            if not global_model_db:
                return None
            
            # Deserialize model
            serialized_model = pickle.loads(global_model_db.model_weights)
            global_model = self.serializer.deserialize_global_model(serialized_model)
            
            return global_model
            
        except Exception as e:
            logger.error(f"Failed to deserialize global model {model_id}: {str(e)}")
            return None
    
    def get_model_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get global model history with metrics."""
        try:
            models = self.list_all(limit=limit)
            
            history = []
            for model in models:
                accuracy_metrics = json.loads(model.accuracy_metrics) if model.accuracy_metrics else {}
                aggregation_info = json.loads(model.aggregation_info) if model.aggregation_info else {}
                
                history.append({
                    'round_number': model.round_number,
                    'version': model.version,
                    'created_at': model.created_at.isoformat(),
                    'model_size_bytes': model.model_size,
                    'accuracy_metrics': accuracy_metrics,
                    'aggregation_info': aggregation_info
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get model history: {str(e)}")
            return []


class ClientRepository(BaseRepository):
    """Repository for client data."""
    
    def __init__(self, session_factory=None):
        """Initialize repository."""
        self.session_factory = session_factory or get_session
    
    def create(self, client_data: Dict[str, Any]) -> Client:
        """Create a new client."""
        try:
            with self.session_factory() as session:
                client = Client(
                    client_id=client_data['client_id'],
                    registration_time=client_data.get('registration_time', datetime.now()),
                    last_seen=client_data.get('last_seen', datetime.now()),
                    status=client_data.get('status', 'active'),
                    capabilities=json.dumps(client_data.get('capabilities', {})),
                    privacy_config=json.dumps(client_data.get('privacy_config', {})),
                    total_updates=client_data.get('total_updates', 0),
                    total_samples=client_data.get('total_samples', 0),
                    client_version=client_data.get('client_version', '1.0.0')
                )
                
                session.add(client)
                session.commit()
                session.refresh(client)
                
                logger.info(f"Created client {client.client_id}")
                return client
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create client: {str(e)}")
            raise
    
    def get_by_id(self, client_id: str) -> Optional[Client]:
        """Get client by ID."""
        try:
            with self.session_factory() as session:
                return session.query(Client).filter(
                    Client.client_id == client_id
                ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get client {client_id}: {str(e)}")
            return None
    
    def update(self, client: Client) -> Client:
        """Update client."""
        try:
            with self.session_factory() as session:
                session.merge(client)
                session.commit()
                logger.debug(f"Updated client {client.client_id}")
                return client
        except SQLAlchemyError as e:
            logger.error(f"Failed to update client: {str(e)}")
            raise
    
    def delete(self, client_id: str) -> bool:
        """Delete client."""
        try:
            with self.session_factory() as session:
                client = session.query(Client).filter(
                    Client.client_id == client_id
                ).first()
                
                if client:
                    session.delete(client)
                    session.commit()
                    logger.info(f"Deleted client {client_id}")
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete client {client_id}: {str(e)}")
            return False
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Client]:
        """List all clients."""
        try:
            with self.session_factory() as session:
                query = session.query(Client).order_by(desc(Client.last_seen))
                
                if limit:
                    query = query.limit(limit).offset(offset)
                
                return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list clients: {str(e)}")
            return []
    
    def get_active_clients(self, minutes: int = 30) -> List[Client]:
        """Get clients active within specified minutes."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            with self.session_factory() as session:
                return session.query(Client).filter(
                    and_(
                        Client.last_seen >= cutoff_time,
                        Client.status == 'active'
                    )
                ).order_by(desc(Client.last_seen)).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get active clients: {str(e)}")
            return []
    
    def update_last_seen(self, client_id: str) -> bool:
        """Update client's last seen timestamp."""
        try:
            with self.session_factory() as session:
                client = session.query(Client).filter(
                    Client.client_id == client_id
                ).first()
                
                if client:
                    client.last_seen = datetime.now()
                    session.commit()
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Failed to update last seen for client {client_id}: {str(e)}")
            return False


class RepositoryManager:
    """Manages all repositories and provides unified access."""
    
    def __init__(self, session_factory=None):
        """Initialize repository manager."""
        self.session_factory = session_factory or get_session
        
        # Initialize repositories
        self.training_rounds = TrainingRoundRepository(session_factory)
        self.client_updates = ClientUpdateRepository(session_factory)
        self.global_models = GlobalModelRepository(session_factory)
        self.clients = ClientRepository(session_factory)
        
        logger.info("RepositoryManager initialized")
    
    def get_training_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        try:
            round_stats = self.training_rounds.get_round_statistics(days)
            active_clients = len(self.clients.get_active_clients())
            latest_model = self.global_models.get_latest()
            
            return {
                'period_days': days,
                'round_statistics': round_stats,
                'active_clients': active_clients,
                'total_registered_clients': len(self.clients.list_all()),
                'latest_model_round': latest_model.round_number if latest_model else 0,
                'latest_model_accuracy': json.loads(latest_model.accuracy_metrics).get('test_accuracy', 0) if latest_model and latest_model.accuracy_metrics else 0
            }
        except Exception as e:
            logger.error(f"Failed to get training summary: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data across all repositories."""
        try:
            results = {
                'client_updates_deleted': self.client_updates.cleanup_old_updates(days_to_keep),
                'old_rounds_found': 0,  # Could implement cleanup for very old rounds
                'old_models_found': 0   # Could implement cleanup for very old models
            }
            
            logger.info(f"Data cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return {'error': str(e)}


# Factory functions
def create_repository_manager(session_factory=None) -> RepositoryManager:
    """Create repository manager instance."""
    return RepositoryManager(session_factory)


def create_training_round_repository(session_factory=None) -> TrainingRoundRepository:
    """Create training round repository instance."""
    return TrainingRoundRepository(session_factory)


def create_client_update_repository(session_factory=None) -> ClientUpdateRepository:
    """Create client update repository instance."""
    return ClientUpdateRepository(session_factory)


def create_global_model_repository(session_factory=None) -> GlobalModelRepository:
    """Create global model repository instance."""
    return GlobalModelRepository(session_factory)


def create_client_repository(session_factory=None) -> ClientRepository:
    """Create client repository instance."""
    return ClientRepository(session_factory)


# Example usage
if __name__ == "__main__":
    # Test repository functionality
    repo_manager = create_repository_manager()
    
    # Test training round creation
    round_data = {
        'round_number': 1,
        'participating_clients': 5,
        'target_clients': 10,
        'configuration': {'epochs': 5, 'batch_size': 32}
    }
    
    training_round = repo_manager.training_rounds.create(round_data)
    print(f"Created training round: {training_round.round_number}")
    
    # Test client creation
    client_data = {
        'client_id': 'test_client_001',
        'capabilities': {'compute_power': 'medium', 'samples': 1000},
        'privacy_config': {'epsilon': 1.0, 'delta': 1e-5}
    }
    
    client = repo_manager.clients.create(client_data)
    print(f"Created client: {client.client_id}")
    
    # Get training summary
    summary = repo_manager.get_training_summary()
    print(f"Training summary: {summary}")