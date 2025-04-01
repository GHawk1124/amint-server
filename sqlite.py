
import sqlite3
import os
import json
import torch
import numpy as np
import uuid
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple, Any, cast

# Database file path
DB_PATH = 'amint.db'

class SQLiteManager:
    """SQLite database manager for handling Google OAuth users and storing memory data"""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the SQLite manager"""
        self.db_path = db_path
        self.initialize_db()
    
    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = 1")
        conn.row_factory = sqlite3.Row
        return conn
    
    def initialize_db(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create users table for Google OAuth
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            picture TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            access_token TEXT,
            refresh_token TEXT,
            token_expiry TIMESTAMP
        )
        ''')
        
        # Create document_containers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_containers (
            document_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            timestamp TIMESTAMP,
            source_type TEXT,
            metadata TEXT,
            user_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # Create memories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            original_text TEXT NOT NULL,
            embeddings BLOB,
            timestamp TIMESTAMP,
            parent_id TEXT,
            section_title TEXT,
            metadata TEXT,
            user_id TEXT,
            FOREIGN KEY (parent_id) REFERENCES document_containers(document_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    # ===== User Management =====
    
    def create_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create a new user from Google OAuth data
        
        Args:
            user_data: Dictionary containing user information from Google OAuth
            
        Returns:
            User ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        user_id = user_data.get('id') or str(uuid.uuid4())
        email = user_data.get('email')
        name = user_data.get('name')
        picture = user_data.get('picture')
        access_token = user_data.get('access_token')
        refresh_token = user_data.get('refresh_token')
        token_expiry = user_data.get('token_expiry')
        
        try:
            cursor.execute('''
            INSERT INTO users (id, email, name, picture, last_login, access_token, refresh_token, token_expiry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                picture = excluded.picture,
                last_login = excluded.last_login,
                access_token = excluded.access_token,
                refresh_token = COALESCE(excluded.refresh_token, users.refresh_token),
                token_expiry = excluded.token_expiry
            ''', (user_id, email, name, picture, datetime.now(), access_token, refresh_token, token_expiry))
            
            conn.commit()
            return user_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User data as dictionary or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        conn.close()
        
        if user:
            return dict(user)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email
        
        Args:
            email: User email
            
        Returns:
            User data as dictionary or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        conn.close()
        
        if user:
            return dict(user)
        return None
    
    def update_user_tokens(self, user_id: str, access_token: str, refresh_token: Optional[str], token_expiry: str) -> bool:
        """
        Update a user's OAuth tokens
        
        Args:
            user_id: User ID
            access_token: New access token
            refresh_token: New refresh token (if available)
            token_expiry: Token expiry timestamp
            
        Returns:
            Success status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if refresh_token:
                cursor.execute('''
                UPDATE users 
                SET access_token = ?, refresh_token = ?, token_expiry = ?, last_login = ?
                WHERE id = ?
                ''', (access_token, refresh_token, token_expiry, datetime.now(), user_id))
            else:
                cursor.execute('''
                UPDATE users 
                SET access_token = ?, token_expiry = ?, last_login = ?
                WHERE id = ?
                ''', (access_token, token_expiry, datetime.now(), user_id))
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    # ===== Document Container Management =====
    
    def create_document_container(self, container_data: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """
        Create a new document container
        
        Args:
            container_data: Dictionary containing document container data
            user_id: Optional user ID to associate this container with
            
        Returns:
            Document container ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        document_id = container_data.get('document_id') or str(uuid.uuid4())
        title = container_data['title']
        timestamp = container_data.get('timestamp') or datetime.now()
        source_type = container_data.get('source_type')
        metadata = json.dumps(container_data.get('metadata') or {})
        
        try:
            cursor.execute('''
            INSERT INTO document_containers (document_id, title, timestamp, source_type, metadata, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (document_id, title, timestamp, source_type, metadata, user_id))
            
            conn.commit()
            return document_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_document_container(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document container by ID
        
        Args:
            document_id: Document container ID
            
        Returns:
            Document container data as dictionary or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM document_containers WHERE document_id = ?', (document_id,))
        container = cursor.fetchone()
        
        conn.close()
        
        if container:
            result = dict(container)
            result['metadata'] = json.loads(result['metadata'])
            return result
        return None
    
    def get_user_document_containers(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all document containers for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of document container data
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM document_containers WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
        containers = cursor.fetchall()
        
        conn.close()
        
        result = []
        for container in containers:
            container_dict = dict(container)
            container_dict['metadata'] = json.loads(container_dict['metadata'])
            result.append(container_dict)
        
        return result
    
    # ===== Memory Management =====
    
    def _blob_to_tensor(self, blob: bytes) -> torch.Tensor:
        """Convert blob to tensor"""
        return torch.tensor(np.frombuffer(blob, dtype=np.float32).reshape(-1), dtype=torch.float32)
    
    def _tensor_to_blob(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to blob"""
        return tensor.detach().cpu().numpy().astype(np.float32).tobytes()
    
    def create_memory(self, memory_data: Dict[str, Any], user_id: Optional[str] = None) -> Tuple[str, bool]:
        """
        Create a new memory or update if memory_id already exists
        Args:
            memory_data: Dictionary containing memory data
            user_id: Optional user ID to associate this memory with
        Returns:
            Tuple of (Memory ID, was_created)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        memory_id = memory_data.get('memory_id') or str(uuid.uuid4())
        name = memory_data['name']
        original_text = memory_data['original_text']
        embeddings = self._tensor_to_blob(memory_data['embeddings'])
        timestamp = memory_data.get('timestamp') or datetime.now()
        parent_id = memory_data.get('parent_id')
        section_title = memory_data.get('section_title')
        metadata = json.dumps(memory_data.get('metadata') or {})
        
        # Check if memory already exists
        cursor.execute('SELECT memory_id FROM memories WHERE memory_id = ?', (memory_id,))
        exists = cursor.fetchone() is not None
        
        try:
            if exists:
                # Update existing memory
                cursor.execute('''
                UPDATE memories SET
                    name = ?,
                    original_text = ?,
                    embeddings = ?,
                    timestamp = ?,
                    parent_id = ?,
                    section_title = ?,
                    metadata = ?,
                    user_id = ?
                WHERE memory_id = ?
                ''', (name, original_text, embeddings, timestamp, parent_id, section_title, metadata, user_id, memory_id))
            else:
                # Insert new memory
                cursor.execute('''
                INSERT INTO memories (memory_id, name, original_text, embeddings, timestamp, parent_id, section_title, metadata, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (memory_id, name, original_text, embeddings, timestamp, parent_id, section_title, metadata, user_id))
            
            conn.commit()
            return memory_id, not exists  # Return ID and whether it was newly created
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get memory by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory data as dictionary or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE memory_id = ?', (memory_id,))
        memory = cursor.fetchone()
        
        conn.close()
        
        if memory:
            memory_dict = dict(memory)
            memory_dict['embeddings'] = self._blob_to_tensor(memory_dict['embeddings'])
            memory_dict['metadata'] = json.loads(memory_dict['metadata'])
            return memory_dict
        return None
    
    def get_document_memories(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of memory data
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE parent_id = ?', (document_id,))
        memories = cursor.fetchall()
        
        conn.close()
        
        result = []
        for memory in memories:
            memory_dict = dict(memory)
            memory_dict['embeddings'] = self._blob_to_tensor(memory_dict['embeddings'])
            memory_dict['metadata'] = json.loads(memory_dict['metadata'])
            result.append(memory_dict)
        
        return result
    
    def get_user_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all memories for a user
        
        Args:
            user_id: User ID
            limit: Maximum number of memories to return
            
        Returns:
            List of memory data
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
        memories = cursor.fetchall()
        
        conn.close()
        
        result = []
        for memory in memories:
            memory_dict = dict(memory)
            memory_dict['embeddings'] = self._blob_to_tensor(memory_dict['embeddings'])
            memory_dict['metadata'] = json.loads(memory_dict['metadata'])
            result.append(memory_dict)
        
        return result
    
    def update_memory(self, memory_id: str, new_data: Dict[str, Any]) -> bool:
        """
        Update an existing memory
        
        Args:
            memory_id: Memory ID
            new_data: Dictionary with fields to update
            
        Returns:
            Success status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Build the update query dynamically based on what's provided
        fields = []
        params = []
        
        if 'name' in new_data:
            fields.append("name = ?")
            params.append(new_data['name'])
        
        if 'original_text' in new_data:
            fields.append("original_text = ?")
            params.append(new_data['original_text'])
        
        if 'embeddings' in new_data:
            fields.append("embeddings = ?")
            params.append(self._tensor_to_blob(new_data['embeddings']))
        
        if 'section_title' in new_data:
            fields.append("section_title = ?")
            params.append(new_data['section_title'])
        
        if 'metadata' in new_data:
            fields.append("metadata = ?")
            params.append(json.dumps(new_data['metadata']))
        
        # Add timestamp and memory_id
        fields.append("timestamp = ?")
        params.append(datetime.now())
        params.append(memory_id)
        
        if not fields:
            return False
        
        try:
            query = f"UPDATE memories SET {', '.join(fields)} WHERE memory_id = ?"
            cursor.execute(query, params)
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Success status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM memories WHERE memory_id = ?', (memory_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def delete_document_container(self, document_id: str, delete_memories: bool = True) -> bool:
        """
        Delete a document container and optionally its memories
        
        Args:
            document_id: Document container ID
            delete_memories: Whether to also delete associated memories
            
        Returns:
            Success status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if delete_memories:
                cursor.execute('DELETE FROM memories WHERE parent_id = ?', (document_id,))
            
            cursor.execute('DELETE FROM document_containers WHERE document_id = ?', (document_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    # ===== Conversion Functions =====
    
    def memory_to_dict(self, memory_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a memory dict from the database to a format compatible with Memory class
        
        Args:
            memory_dict: Memory dict from database
            
        Returns:
            Dict formatted for Memory class
        """
        return {
            'memory_id': memory_dict['memory_id'],
            'name': memory_dict['name'],
            'original_text': memory_dict['original_text'],
            'embeddings': memory_dict['embeddings'],  # Already converted to tensor
            'timestamp': memory_dict['timestamp'],
            'parent_id': memory_dict['parent_id'],
            'section_title': memory_dict['section_title'],
            'metadata': memory_dict['metadata']
        }
    
    def container_to_dict(self, container_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a container dict from the database to a format compatible with DocumentContainer class
        
        Args:
            container_dict: Container dict from database
            
        Returns:
            Dict formatted for DocumentContainer class
        """
        return {
            'document_id': container_dict['document_id'],
            'title': container_dict['title'],
            'timestamp': container_dict['timestamp'],
            'source_type': container_dict['source_type'],
            'metadata': container_dict['metadata']
        }

# Example for how to use this with ModernHopfieldNetwork from temporal_hopfield_network.py
def create_hopfield_network_from_db(db_manager: SQLiteManager, document_id: str = None, user_id: str = None):
    """
    Create a ModernHopfieldNetwork and load memories from the database
    
    Args:
        db_manager: SQLiteManager instance
        document_id: Optional document ID to load memories from
        user_id: Optional user ID to load memories from
        
    Returns:
        ModernHopfieldNetwork with loaded memories
    """
    from temporal_hopfield_network import ModernHopfieldNetwork, Memory, DocumentContainer
    
    # Get embedding dimensions from first memory (assuming at least one exists)
    if document_id:
        memories_data = db_manager.get_document_memories(document_id)
    elif user_id:
        memories_data = db_manager.get_user_memories(user_id)
    else:
        raise ValueError("Either document_id or user_id must be provided")
    
    if not memories_data:
        # Default dimension if no memories exist
        embedding_dim = 384  # Default for 'all-MiniLM-L6-v2' model
    else:
        embedding_dim = memories_data[0]['embeddings'].shape[0]
    
    # Create network
    network = ModernHopfieldNetwork(embedding_dim=embedding_dim)
    
    # Load document containers if needed
    if document_id:
        container_data = db_manager.get_document_container(document_id)
        if container_data:
            container = DocumentContainer(
                title=container_data['title'],
                document_id=container_data['document_id'],
                timestamp=container_data['timestamp'],
                source_type=container_data['source_type'],
                metadata=container_data['metadata']
            )
            network.add_document_container(container)
    
    # Load memories
    for memory_data in memories_data:
        memory = Memory(
            name=memory_data['name'],
            original_text=memory_data['original_text'],
            embeddings=memory_data['embeddings'],
            memory_id=memory_data['memory_id'],
            timestamp=memory_data['timestamp'],
            parent_id=memory_data['parent_id'],
            section_title=memory_data['section_title'],
            metadata=memory_data['metadata']
        )
        network.store(memory)
    
    return network

def save_network_to_db(network: Any, db_manager: SQLiteManager, user_id: Optional[str] = None):
    """
    Save a ModernHopfieldNetwork to the database
    Args:
        network: ModernHopfieldNetwork instance
        db_manager: SQLiteManager instance
        user_id: Optional user ID to associate with memories
    Returns:
        Dict with statistics about created and updated objects
    """
    from temporal_hopfield_network import ModernHopfieldNetwork
    network = cast(ModernHopfieldNetwork, network)
    
    stats = {
        "documents_created": 0,
        "documents_updated": 0,
        "memories_created": 0,
        "memories_updated": 0
    }
    
    # Save document containers
    for doc_id, container in network.document_containers.items():
        container_data = {
            'document_id': container.document_id,
            'title': container.title,
            'timestamp': container.timestamp,
            'source_type': container.source_type,
            'metadata': container.metadata
        }
        try:
            # Check if document exists first
            existing_doc = db_manager.get_document_container(container.document_id)
            if existing_doc:
                # Update logic would go here if needed
                stats["documents_updated"] += 1
            else:
                db_manager.create_document_container(container_data, user_id)
                stats["documents_created"] += 1
        except Exception as e:
            print(f"Warning: Could not save document container {container.document_id}: {str(e)}")
    
    # Save memories
    for memory in network.memories:
        memory_data = {
            'memory_id': memory.memory_id,
            'name': memory.name,
            'original_text': memory.original_text,
            'embeddings': memory.embeddings,
            'timestamp': memory.timestamp,
            'parent_id': memory.parent_id,
            'section_title': memory.section_title,
            'metadata': memory.metadata
        }
        try:
            # Check if memory exists before attempting to save
            existing_memory = db_manager.get_memory(memory.memory_id)
            if existing_memory:
                # Update existing memory
                db_manager.update_memory(memory.memory_id, memory_data)
                stats["memories_updated"] += 1
            else:
                # Create new memory
                db_manager.create_memory(memory_data, user_id)
                stats["memories_created"] += 1
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                # Handle unique constraint failure by updating instead
                db_manager.update_memory(memory.memory_id, memory_data)
                stats["memories_updated"] += 1
            else:
                raise
        except Exception as e:
            print(f"Warning: Could not save memory {memory.memory_id}: {str(e)}")
    
    return stats