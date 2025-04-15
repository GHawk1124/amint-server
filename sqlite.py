
import sqlite3
import os
import json
import torch
import numpy as np
import uuid
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple, Any, cast
from temporal_hopfield_network import ModernHopfieldNetwork, Memory, DocumentContainer
from khm_network import KernelizedHopfieldNetwork, FeatureMap

DEFAULT_KHM_FEATURE_DIM = 384
DEFAULT_KHM_HIDDEN_DIM = 512
DEFAULT_KHM_NUM_HEADS = 4
DEFAULT_EMBEDDING_DIM = 384

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

    def update_document_container(self, document_id: str, data_to_update: Dict[str, Any]) -> bool:
        """Update specific fields of a document container."""
        conn = self.get_connection()
        cursor = conn.cursor()
        fields = []
        params = []
        allowed_fields = ['title', 'source_type', 'metadata', 'timestamp'] # Fields allowed to be updated

        for field in allowed_fields:
            if field in data_to_update:
                fields.append(f"{field} = ?")
                value = data_to_update[field]
                # Serialize metadata
                if field == 'metadata':
                    value = json.dumps(value or {})
                params.append(value)

        if not fields:
            return False # Nothing to update

        params.append(document_id) # For the WHERE clause

        try:
            query = f"UPDATE document_containers SET {', '.join(fields)} WHERE document_id = ?"
            cursor.execute(query, tuple(params))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            print(f"Error updating document container {document_id}: {e}")
            raise e
        finally:
            conn.close()
            
    # Add count_user_document_containers
    def count_user_document_containers(self, user_id: str) -> int:
         """Count all document containers for a user."""
         conn = self.get_connection()
         cursor = conn.cursor()
         cursor.execute('SELECT COUNT(*) FROM document_containers WHERE user_id = ?', (user_id,))
         count = cursor.fetchone()[0]
         conn.close()
         return count

    # --- MODIFIED: Add optional document_id parameter ---
    def count_user_memories(self, user_id: str, document_id: Optional[str] = None) -> int:
         """Count memories for a user, optionally filtered by document."""
         conn = self.get_connection()
         cursor = conn.cursor()
         count = 0 # Initialize count
         try:
             if document_id:
                 # Ensure the user owns the document before counting its memories
                 # Check ownership first
                 cursor.execute('SELECT user_id FROM document_containers WHERE document_id = ?', (document_id,))
                 container_owner = cursor.fetchone()

                 is_dev_null_access = user_id == "dev_user_123" and container_owner and container_owner['user_id'] is None

                 if container_owner and (container_owner['user_id'] == user_id or is_dev_null_access):
                      cursor.execute('SELECT COUNT(*) FROM memories WHERE parent_id = ?', (document_id,))
                      count_result = cursor.fetchone()
                      count = count_result[0] if count_result else 0
                 else:
                      # Document doesn't exist or user doesn't own it
                      print(f"[Info] count_user_memories: Document {document_id} not found or not owned by user {user_id}.")
                      count = 0
             else:
                 # Count all memories for the user
                 cursor.execute('SELECT COUNT(*) FROM memories WHERE user_id = ?', (user_id,))
                 count_result = cursor.fetchone()
                 count = count_result[0] if count_result else 0
         except Exception as e:
              print(f"Error in count_user_memories: {e}")
              count = 0 # Return 0 on error
         finally:
              conn.close()
         return count
    
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
    
    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Alias for get_user_document_containers to maintain API compatibility
        Args:
            user_id: User ID
        Returns:
            List of document container data
        """
        return self.get_user_document_containers(user_id)
    
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
    
    def count_user_memories(self, user_id: str, document_id: Optional[str] = None) -> int:
         """Count memories for a user, optionally filtered by document."""
         conn = self.get_connection()
         cursor = conn.cursor()
         count = 0 # Initialize count
         try:
             if document_id:
                 # Ensure the user owns the document before counting its memories
                 # Check ownership first
                 cursor.execute('SELECT user_id FROM document_containers WHERE document_id = ?', (document_id,))
                 container_owner = cursor.fetchone()
                 # Allow dev user access if container user_id is NULL
                 is_dev_null_access = user_id == "dev_user_123" and container_owner and container_owner['user_id'] is None
                 if container_owner and (container_owner['user_id'] == user_id or is_dev_null_access):
                      cursor.execute('SELECT COUNT(*) FROM memories WHERE parent_id = ?', (document_id,))
                      count_result = cursor.fetchone()
                      count = count_result[0] if count_result else 0
                 else:
                      # Document doesn't exist or user doesn't own it
                      print(f"[Info] count_user_memories: Document {document_id} not found or not owned by user {user_id}.")
                      count = 0
             else:
                 # Count all memories for the user
                 cursor.execute('SELECT COUNT(*) FROM memories WHERE user_id = ?', (user_id,))
                 count_result = cursor.fetchone()
                 count = count_result[0] if count_result else 0
         except Exception as e:
              print(f"Error in count_user_memories: {e}")
              count = 0 # Return 0 on error
         finally:
              conn.close()
         return count
    
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

def create_hopfield_network_from_db(
    db_manager: SQLiteManager,
    network_type: str, # <-- NEW: Specify which type to create
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
    khm_config: Optional[Dict[str, Any]] = None # <-- NEW: KHM parameters
) -> Union[ModernHopfieldNetwork, KernelizedHopfieldNetwork, Any]:
    """
    Create a Hopfield Network (Modern or KHM) and load memories from the database.
    Args:
        db_manager: SQLiteManager instance.
        network_type: 'modern' or 'khm'.
        document_id: Optional document ID to load memories from.
        user_id: Optional user ID to load memories from.
        khm_config: Optional dict with 'feature_dim', 'hidden_dim', 'num_heads' for KHM.
    Returns:
        Initialized Hopfield network with loaded memories.
    """
    if not user_id:
         # If only document_id is provided, we need to find the user_id associated with it
         if document_id:
              container_data = db_manager.get_document_container(document_id)
              if container_data and container_data.get('user_id'):
                   user_id = container_data['user_id']
              else:
                   # Handle cases where document has no user or doesn't exist
                   # If it's a shared/unowned document, maybe load without user context?
                   # For now, raise error if user context is expected but not found
                   raise ValueError(f"Cannot determine user_id for document_id {document_id}")
         else:
              raise ValueError("Either document_id or user_id must be provided")

    print(f"Loading network data for user {user_id} (Doc: {document_id}, Type: {network_type.upper()})")

    # Determine memories to load
    if document_id:
        # Important: Verify user owns this document before loading memories
        container_data = db_manager.get_document_container(document_id)
        if not container_data or container_data.get('user_id') != user_id:
             # Allow dev user access to null user_id docs?
             is_dev_accessing_null = user_id == "dev_user_123" and container_data and container_data.get('user_id') is None
             if not is_dev_accessing_null:
                  raise ValueError(f"User {user_id} not authorized for document {document_id}")
        memories_data = db_manager.get_document_memories(document_id)
    else:
        # Load all memories for the user
        memories_data = db_manager.get_user_memories(user_id, limit=10000) # Adjust limit as needed

    # Determine embedding dimension
    if not memories_data:
        embedding_dim = DEFAULT_EMBEDDING_DIM
        print(f"No memories found in DB for this scope. Using default embedding dim: {embedding_dim}")
    else:
        # Ensure embeddings are valid tensors before accessing shape
        first_valid_embedding = next((m['embeddings'] for m in memories_data if isinstance(m.get('embeddings'), torch.Tensor)), None)
        if first_valid_embedding is not None:
             embedding_dim = first_valid_embedding.shape[0]
             print(f"Determined embedding dimension from loaded memories: {embedding_dim}")
        else:
             embedding_dim = DEFAULT_EMBEDDING_DIM
             print(f"[Warning] No valid tensor embeddings found in loaded data. Using default dim: {embedding_dim}")


    # Create the appropriate network instance
    network: Union[ModernHopfieldNetwork, KernelizedHopfieldNetwork]
    if network_type == "khm":
        cfg = khm_config or {}
        feature_dim = cfg.get("feature_dim", embedding_dim) # Default feature dim to embedding dim
        hidden_dim = cfg.get("hidden_dim", DEFAULT_KHM_HIDDEN_DIM)
        num_heads = cfg.get("num_heads", DEFAULT_KHM_NUM_HEADS)
        print(f"Instantiating KernelizedHopfieldNetwork (Embed: {embedding_dim}, Feature: {feature_dim}, Hidden: {hidden_dim}, Heads: {num_heads})")
        network = KernelizedHopfieldNetwork(
            embedding_dim=embedding_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        # --- TODO: Load KHM Kernel State ---
        # This requires saving/loading the feature_map's state_dict,
        # potentially as a separate file or blob linked to the user/network.
        # kernel_state_path = f"user_{user_id}_khm_kernel.pth"
        # if os.path.exists(kernel_state_path):
        #     try:
        #         network.feature_map.load_state_dict(torch.load(kernel_state_path))
        #         network.feature_map.eval() # Set to evaluation mode
        #         print(f"Loaded trained KHM kernel state from {kernel_state_path}")
        #     except Exception as load_e:
        #         print(f"[Warning] Failed to load KHM kernel state: {load_e}. Kernel will be untrained.")
        # else:
        #     print("No saved KHM kernel state found. Kernel is untrained.")

    else: # Default to Modern
        print(f"Instantiating ModernHopfieldNetwork (Embed: {embedding_dim})")
        network = ModernHopfieldNetwork(
            embedding_dim=embedding_dim,
            beta=8.0 # Or load beta from config if needed
        )

    # Load document containers associated with the loaded memories or the user
    print("Loading document containers...")
    relevant_container_ids = set(m['parent_id'] for m in memories_data if m.get('parent_id'))
    if document_id:
         relevant_container_ids.add(document_id) # Ensure the specified one is included

    # Fetch containers either specifically requested or belonging to the user
    containers_to_load = {}
    if document_id and document_id in relevant_container_ids:
         container_data = db_manager.get_document_container(document_id)
         if container_data:
              containers_to_load[document_id] = container_data
    else: # Load all user containers if no specific doc ID was the primary filter
         user_containers = db_manager.get_user_document_containers(user_id)
         for container_data in user_containers:
              containers_to_load[container_data['document_id']] = container_data

    for doc_id, container_data in containers_to_load.items():
         # Filter for authorized containers again (belt-and-suspenders)
         owner_id = container_data.get('user_id')
         is_dev_null_access = user_id == "dev_user_123" and owner_id is None
         if owner_id == user_id or is_dev_null_access:
             container = DocumentContainer(
                 title=container_data['title'], document_id=container_data['document_id'],
                 timestamp=container_data['timestamp'], source_type=container_data['source_type'],
                 metadata=container_data['metadata']
             )
             network.add_document_container(container)
         else:
              print(f"[Warning] Skipping unauthorized container {doc_id} during network load.")

    print(f"Loaded {len(network.document_containers)} document containers into network object.")


    # Load memories into the network object
    loaded_memory_count = 0
    for memory_data in memories_data:
         # Ensure embedding is a tensor before creating Memory object
         if not isinstance(memory_data.get('embeddings'), torch.Tensor):
              print(f"[Warning] Skipping memory {memory_data.get('memory_id')} due to invalid embedding type: {type(memory_data.get('embeddings'))}")
              continue

         memory = Memory(
             name=memory_data['name'], original_text=memory_data['original_text'],
             embeddings=memory_data['embeddings'], memory_id=memory_data['memory_id'],
             timestamp=memory_data['timestamp'], parent_id=memory_data['parent_id'],
             section_title=memory_data['section_title'], metadata=memory_data['metadata']
         )
         # Add memory to network state and update container link
         network.store(memory)
         loaded_memory_count += 1

    print(f"Loaded {loaded_memory_count} memories into network object.")

    if not network.memories:
        print(f"[Info] No memories loaded into network object for user {user_id} (Doc: {document_id}). Network is empty.")


    return network

# --- MODIFIED: Function to save network to DB ---
def save_network_to_db(
    network: Union[ModernHopfieldNetwork, KernelizedHopfieldNetwork, Any],
    db_manager: SQLiteManager,
    user_id: Optional[str] = None
) -> Dict[str, int]:
    """
    Save a Hopfield Network (memories and containers) to the database.
    Args:
        network: ModernHopfieldNetwork or KernelizedHopfieldNetwork instance.
        db_manager: SQLiteManager instance.
        user_id: Optional user ID to associate with memories/containers.
    Returns:
        Dict with statistics about created and updated objects.
    """
    if not user_id:
        print("[Warning] Saving network without a user_id. Objects may not be linked to a user.")
        # Potentially try to infer user_id if all objects belong to one user? Risky.

    stats = {
        "documents_created": 0, "documents_updated": 0, "documents_failed": 0,
        "memories_created": 0, "memories_updated": 0, "memories_failed": 0
    }

    # Save/Update document containers present in the network object
    for doc_id, container in network.document_containers.items():
        container_data = {
            'document_id': container.document_id, 'title': container.title,
            'timestamp': container.timestamp, 'source_type': container.source_type,
            'metadata': container.metadata
        }
        try:
            existing_doc = db_manager.get_document_container(container.document_id)
            if existing_doc:
                # Update existing document - ensure user_id matches if already set
                if existing_doc.get('user_id') and existing_doc.get('user_id') != user_id:
                     print(f"[Warning] Skipping update for document container {container.document_id} due to user_id mismatch.")
                     stats["documents_failed"] += 1
                     continue
                db_manager.update_document_container(container.document_id, container_data)
                stats["documents_updated"] += 1
            else:
                # Create new document
                db_manager.create_document_container(container_data, user_id)
                stats["documents_created"] += 1
        except Exception as e:
            print(f"[Error] Failed to save/update document container {container.document_id}: {str(e)}")
            stats["documents_failed"] += 1

    # Save/Update memories present in the network object
    for memory in network.memories:
        # Ensure embedding is a tensor
        if not isinstance(memory.embeddings, torch.Tensor):
             print(f"[Warning] Skipping memory {memory.memory_id} during save: Invalid embedding type {type(memory.embeddings)}")
             stats["memories_failed"] += 1
             continue

        memory_data = {
            'memory_id': memory.memory_id, 'name': memory.name,
            'original_text': memory.original_text, 'embeddings': memory.embeddings,
            'timestamp': memory.timestamp, 'parent_id': memory.parent_id,
            'section_title': memory.section_title, 'metadata': memory.metadata
        }
        try:
            # Use create_memory which handles INSERT OR UPDATE logic based on existence check
            _, created = db_manager.create_memory(memory_data, user_id)
            if created:
                stats["memories_created"] += 1
            else:
                stats["memories_updated"] += 1
        except Exception as e:
            print(f"[Error] Failed to save/update memory {memory.memory_id}: {str(e)}")
            stats["memories_failed"] += 1


    # --- TODO: Save KHM Kernel State ---
    # if isinstance(network, KernelizedHopfieldNetwork) and user_id:
    #     kernel_state_path = f"user_{user_id}_khm_kernel.pth"
    #     try:
    #         torch.save(network.feature_map.state_dict(), kernel_state_path)
    #         print(f"Saved KHM kernel state to {kernel_state_path}")
    #     except Exception as save_e:
    #         print(f"[Error] Failed to save KHM kernel state: {save_e}")

    return stats