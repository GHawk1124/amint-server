from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import json
import uuid
import httpx
from datetime import datetime, timedelta
import shutil
import asyncio
from pathlib import Path
import torch
import numpy as np
import traceback
from google.generativeai import GenerativeModel
import google.generativeai as genai
from gemini import format_context_from_memories, query_gemini
# Import our modules
from temporal_hopfield_network import ModernHopfieldNetwork, Memory, DocumentContainer, TextProcessor
from khm_network import KernelizedHopfieldNetwork
from sqlite import SQLiteManager, create_hopfield_network_from_db, save_network_to_db
from mistral import get_ocr_result
from dotenv import load_dotenv
import jwt
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware
import hashlib
from fastapi import Path as FastApiPath

# Function to calculate file hash
def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Function to check if a file hash already exists
def file_hash_exists(db_manager, user_id, file_hash):
    """Check if a file with this hash has already been uploaded by this user"""
    try:
        # Get all documents for this user
        documents = db_manager.get_user_document_containers(user_id)
        # Check if any document has this hash in its metadata
        for doc in documents:
            metadata = doc['metadata'] or {}
            if metadata.get('file_hash') == file_hash:
                return True, doc
        return False, None
    except Exception as e:
        print(f"Error checking file hash: {str(e)}")
        return False, None

# Load environment variables
load_dotenv()
ENV = os.getenv("APP_ENV", "development")
print(ENV)
IS_PRODUCTION = ENV == "production"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise ValueError("Google OAuth environment variables not set")

NETWORK_TYPE = os.getenv("NETWORK_TYPE", "modern").lower() # Default to 'modern'
# --- NEW: KHM Configuration (Optional - Read from env or use defaults) ---
KHM_FEATURE_DIM = int(os.getenv("KHM_FEATURE_DIM", "384")) # Default to embedding_dim
KHM_HIDDEN_DIM = int(os.getenv("KHM_HIDDEN_DIM", "512"))
KHM_NUM_HEADS = int(os.getenv("KHM_NUM_HEADS", "4"))
KHM_BETA = float(os.getenv("KHM_BETA", "8.0")) # Beta for KHM retrieval
KHM_TRAIN_EPOCHS = int(os.getenv("KHM_TRAIN_EPOCHS", "300"))
KHM_TRAIN_LR = float(os.getenv("KHM_TRAIN_LR", "0.001"))
DEFAULT_EMBEDDING_DIM = 384 # Default for MiniLM-L6-v2 if no memories exist yet

# Define context limits (e.g., in characters)
# Example: Aim for ~16k char (~4k token) context window, use half for buffer
DEFAULT_CONTEXT_CHAR_LIMIT = 16000
MAX_CONTEXT_CHAR_TARGET = int(os.getenv("MAX_CONTEXT_CHAR_TARGET", DEFAULT_CONTEXT_CHAR_LIMIT))
CONTEXT_BUDGET = MAX_CONTEXT_CHAR_TARGET // 2
print(f"Using Context Budget (Chars): {CONTEXT_BUDGET} (Target Window: {MAX_CONTEXT_CHAR_TARGET})")

print(f"Using Network Type: {NETWORK_TYPE.upper()}")
if NETWORK_TYPE == "khm":
    print(f"  KHM Config: feature_dim={KHM_FEATURE_DIM}, hidden_dim={KHM_HIDDEN_DIM}, heads={KHM_NUM_HEADS}, beta={KHM_BETA}")

# Session secret key
SECRET_KEY = os.getenv("SECRET_KEY", str(uuid.uuid4()))

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the FastAPI app
app = FastAPI(title="AMINT API", description="API for the AMINT Temporal Hopfield Network")

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Production allowed origins
PRODUCTION_ORIGINS = [
    "https://GHawk1124.github.io",
    "https://ghawk1124.github.io",
    "https://www.ghawk1124.github.io",
    "https://amint-server.onrender.com",
    "https://amintserver.share.zrok.io",
]

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=PRODUCTION_ORIGINS if IS_PRODUCTION else [
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Initialize SQLite manager
db_manager = SQLiteManager()

# Initialize text processor
text_processor = TextProcessor()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# download nltk
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Data models
class UserLogin(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    access_token: str
    refresh_token: Optional[str] = None
    token_expiry: Optional[str] = None

class DocumentRequest(BaseModel):
    title: str
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class NetworkInfo(BaseModel):
    user_id: str
    document_id: Optional[str] = None

class MemoryQuery(BaseModel):
    query_text: str
    document_id: Optional[str] = None
    similarity_threshold: float = 0.75 # Use threshold instead of k, with a default
    use_gemini: bool = False
    max_gemini_context: int = 5 # Max memories to feed to Gemini

class FileQuery(BaseModel):
    query_text: str
    file_types: Optional[List[str]] = None
    # k: int = 5 # Remove k
    similarity_threshold: float = 0.7 # Add threshold for file search? Default lower?
    # Or keep k for file search if thresholding doesn't make sense? Let's keep k for now unless specified otherwise.
    k: int = 5

class DummyFileData(BaseModel):
    file_count: int = 5  # Number of dummy files to generate
    user_id: str  # User ID to add the files to

class MemoryQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    gemini_response: Optional[str] = None
    use_gemini: bool = False
    retrieved_count: int # Add count of retrieved items before Gemini limit
    threshold_used: float # Add threshold used
    similarity_scores: Optional[List[float]] = None

# OAuth2 password bearer for token validation
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Helper function to get user from session
async def get_current_user(request: Request):
    """Get the current user from session"""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db_manager.get_user(user_id)

def get_user_network(user_id: str, document_id: Optional[str] = None) -> Union[ModernHopfieldNetwork, KernelizedHopfieldNetwork, Any]: # Use Union or Any
    """Get or create a Hopfield network (Modern or KHM) for a user based on NETWORK_TYPE."""
    global NETWORK_TYPE # Access the global config
    try:
        # Try to load existing network from database, passing the type
        network = create_hopfield_network_from_db(
            db_manager,
            network_type=NETWORK_TYPE,
            document_id=document_id,
            user_id=user_id,
            # Pass KHM config in case loading needs it (though create_hopfield currently determines dim from data)
            khm_config={
                "feature_dim": KHM_FEATURE_DIM,
                "hidden_dim": KHM_HIDDEN_DIM,
                "num_heads": KHM_NUM_HEADS
            }
        )
        # TODO: If KHM, potentially load kernel state here if sqlite.py is enhanced
        return network
    except Exception as e:
        # If no network exists or loading fails, create a new one
        print(f"Creating new {NETWORK_TYPE.upper()} network for user {user_id}. Reason: {str(e)}")
        if NETWORK_TYPE == "khm":
            # Pass KHM specific config
            network = KernelizedHopfieldNetwork(
                embedding_dim=DEFAULT_EMBEDDING_DIM, # Default, might be overwritten if data exists later
                feature_dim=KHM_FEATURE_DIM,
                hidden_dim=KHM_HIDDEN_DIM,
                num_heads=KHM_NUM_HEADS
            )
        else: # Default to Modern
            network = ModernHopfieldNetwork(
                embedding_dim=DEFAULT_EMBEDDING_DIM, # Default
                beta=8.0 # Or read beta from env if needed for ModernHopfieldNetwork too
            )
        return network

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to AMINT API"}

# Google OAuth routes
@app.get("/auth/google/login")
async def google_login():
    """Start the Google OAuth flow"""
    google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",  # Force refresh token
    }
    url = f"{google_auth_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    return RedirectResponse(url)

@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str):
    """Handle the Google OAuth callback"""
    try:
        # Exchange authorization code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": GOOGLE_REDIRECT_URI,
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(token_url, data=token_data)
            
        if not token_response.status_code == 200:
            raise HTTPException(status_code=400, detail="Failed to get token from Google")
            
        token_info = token_response.json()
        access_token = token_info.get("access_token")
        refresh_token = token_info.get("refresh_token")
        id_token = token_info.get("id_token")
        
        # Decode the id_token to get user info (without verification for simplicity)
        # In production, you should verify the token's signature
        # Different JWT libraries have different methods - this works with PyJWT
        try:
            # First, try the PyJWT style
            user_info = jwt.decode(id_token, key=None, algorithms=["RS256"], options={"verify_signature": False})
        except (AttributeError, TypeError):
            # If that fails, try to parse it manually as a fallback
            import base64
            import json
            
            # Extract the payload part of the JWT (second part)
            parts = id_token.split('.')
            if len(parts) != 3:
                raise ValueError("Invalid JWT format")
                
            # Get the middle part (payload)
            payload = parts[1]
            
            # Add padding if needed
            payload += '=' * (4 - len(payload) % 4)
            
            # Decode base64
            decoded = base64.b64decode(payload.replace('-', '+').replace('_', '/'))
            
            # Parse JSON
            user_info = json.loads(decoded)
        
        # Create a unique ID for this Google user
        google_user_id = f"google_{user_info.get('sub')}"
        
        # Create/update user in database
        user_data = {
            "id": google_user_id,
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture"),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_expiry": str(datetime.now() + timedelta(seconds=token_info.get("expires_in", 3600)))
        }
        
        user_id = db_manager.create_user(user_data)
        
        # Save user ID in session
        request.session["user_id"] = user_id
        
        # Create initial empty network if user is new
        network = get_user_network(user_id)
        
        # Redirect to the frontend with success
        frontend_url = "http://localhost:1420" if not IS_PRODUCTION else "https://ghawk1124.github.io/amint/dist"
        return RedirectResponse(f"{frontend_url}?auth=success")
    
    except Exception as e:
        print(f"Google OAuth error: {str(e)}")
        print(traceback.format_exc())
        # Redirect to the frontend with error
        frontend_url = "http://localhost:1420" if not IS_PRODUCTION else "https://ghawk1124.github.io/amint/dist"
        return RedirectResponse(f"{frontend_url}?auth=error&message={str(e)}")

@app.get("/auth/session")
async def get_session(request: Request):
    """Get the current user session"""
    user = await get_current_user(request)
    if not user:
        return {"authenticated": False}
    
    # Remove sensitive information
    if "access_token" in user:
        del user["access_token"]
    if "refresh_token" in user:
        del user["refresh_token"]
    if "token_expiry" in user:
        del user["token_expiry"]
        
    return {
        "authenticated": True,
        "user": user
    }

@app.post("/auth/logout")
async def logout(request: Request):
    """Log out the current user"""
    request.session.clear()
    return {"status": "success"}

if not IS_PRODUCTION:
    # New simple development login endpoint
    @app.post("/auth/dev-login")
    async def dev_login(request: Request):
        """Development-only endpoint for quick login without OAuth"""
        try:
            # Create a fixed development user for testing
            dev_user = {
                "id": "dev_user_123",
                "email": "dev@example.com",
                "name": "Dev User",
                "picture": "https://via.placeholder.com/150",
                "access_token": "dev_token"
            }
            user_id = db_manager.create_user(dev_user)
            
            # Save user ID in session
            request.session["user_id"] = user_id
            
            # Create initial empty network if user is new
            network = get_user_network(user_id)
            
            return {"user_id": user_id, "email": dev_user["email"], "name": dev_user["name"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Development login failed: {str(e)}")

@app.get("/users/me")
async def get_current_user_info(request: Request):
    """Get the current authenticated user's information"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    # Remove sensitive information
    if "access_token" in user:
        del user["access_token"]
    if "refresh_token" in user:
        del user["refresh_token"]
    if "token_expiry" in user:
        del user["token_expiry"]
        
    return user

@app.get("/users/{user_id}")
async def get_user(user_id: str, request: Request):
    """Get user information (checking session authorization)"""
    current_user = await get_current_user(request)
    
    # Basic authorization check - only allow getting your own user info or admin
    if not current_user or (current_user.get("id") != user_id and not current_user.get("is_admin", False)):
        raise HTTPException(status_code=403, detail="Not authorized to access this user information")
    
    user = db_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Remove sensitive information
    if "access_token" in user:
        del user["access_token"]
    if "refresh_token" in user:
        del user["refresh_token"]
    if "token_expiry" in user:
        del user["token_expiry"]
        
    return user

@app.post("/documents/create")
async def create_document(doc_request: DocumentRequest, request: Request):
    """Create a new document container"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        # Create document container
        container = DocumentContainer(
            title=doc_request.title,
            source_type=doc_request.source_type,
            metadata=doc_request.metadata
        )
        # Store in database
        db_manager.add_document_container(container, user_id)
        return {
            "document_id": container.document_id,
            "title": container.title,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")

@app.get("/documents")
async def list_documents(request: Request):
    """List all documents for a user"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        documents = db_manager.get_user_documents(user_id)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.post("/documents/upload")
async def upload_document(
    request: Request,
    document_id: Optional[str] = Form(None),
    title: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a document file, process it, add to the network, and potentially train KHM kernel."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    network: Union[ModernHopfieldNetwork, KernelizedHopfieldNetwork, Any] = None # Initialize network variable
    file_path_str = None # Initialize file_path_str

    try:
        user_id = user["id"]
        # --- File Handling & Hashing (Keep as is) ---
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        file_path_str = str(file_path) # Store as string for finally block

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_hash = calculate_file_hash(file_path_str)
        hash_exists, existing_doc = file_hash_exists(db_manager, user_id, file_hash)
        if hash_exists:
            # Clean up the newly uploaded (duplicate) file before returning
            os.remove(file_path_str)
            return {
                "status": "duplicate",
                "message": "This file has already been uploaded",
                "document_id": existing_doc['document_id'],
                "title": existing_doc['title'],
                "original_upload_date": existing_doc['timestamp']
            }

        # --- OCR / Text Extraction (Keep as is) ---
        if file_extension in ['.pdf', '.jpg', '.jpeg', '.png']:
            extracted_text = get_ocr_result(MISTRAL_API_KEY, file_path_str)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()

        # --- Document Container Creation/Update (Keep as is, but ensure doc ID is captured) ---
        container_metadata = {
            "original_filename": file.filename,
            "file_hash": file_hash,
            "upload_date": str(datetime.now())
        }
        if not document_id:
            container = DocumentContainer(
                title=title,
                source_type=f"{file_extension.lstrip('.')}_uploaded_file", # More specific source type
                metadata=container_metadata
            )
            container_data = {
                'document_id': container.document_id, 'title': container.title,
                'timestamp': container.timestamp, 'source_type': container.source_type,
                'metadata': container.metadata
            }
            document_id = db_manager.create_document_container(container_data, user_id)
            print(f"Created document container with ID: {document_id}")
        else:
            # Verify existing container and update hash
            existing_container_data = db_manager.get_document_container(document_id)
            if not existing_container_data:
                 raise HTTPException(status_code=404, detail="Document container not found")
            # Important: Ensure user owns this container
            if existing_container_data.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Not authorized to add to this document container")

            existing_meta = existing_container_data.get('metadata', {})
            existing_meta.update(container_metadata) # Merge new meta like hash
            # Update the title and metadata in the DB if necessary
            db_manager.update_document_container(document_id, {'title': title, 'metadata': existing_meta})
            print(f"Using existing document container ID: {document_id}")


        # --- Text Processing (Keep as is) ---
        container_for_processing, memories = text_processor.process_document(
            title=title,
            text=extracted_text,
            chunk_size=3,
            max_chunk_length=1000,
            use_section_detection=True
        )

        # --- IMPORTANT: Assign the correct document_id to memories ---
        for memory in memories:
            memory.parent_id = document_id # Ensure memories link to the correct container

        # --- Network Interaction ---
        # Get the correct network type (Modern or KHM)
        # Pass the specific document_id to load only relevant parts if applicable by implementation
        network = get_user_network(user_id, document_id=None) # Load the user's general network first

        # Ensure network's embedding dimension matches if it was newly created
        if not network.memories and memories: # If network is new and we have memories
            first_embedding_dim = memories[0].embeddings.shape[0]
            if network.embedding_dim != first_embedding_dim:
                print(f"Adjusting new network embedding dimension from {network.embedding_dim} to {first_embedding_dim}")
                network.embedding_dim = first_embedding_dim
                # If KHM, the feature map input dim might need adjustment IF it wasn't set correctly initially
                if NETWORK_TYPE == 'khm' and isinstance(network, KernelizedHopfieldNetwork):
                     # Re-initialize feature map if necessary (this loses untrained state)
                     # A better approach would be to ensure correct initial dim
                     if network.feature_map.layer1.in_features != first_embedding_dim:
                          print(f"Re-initializing KHM feature map for input dim {first_embedding_dim}")
                          network.feature_map = FeatureMap(first_embedding_dim, KHM_HIDDEN_DIM, KHM_FEATURE_DIM)

        # Add memories to network
        network.batch_store(memories)

        # --- Save Network State ---
        # TODO: Enhance save_network_to_db to handle KHM kernel state if needed
        save_stats = save_network_to_db(network, db_manager, user_id)
        print(f"Network save stats: {save_stats}")

        # --- KHM Specific: Train Kernel ---
        if NETWORK_TYPE == "khm" and isinstance(network, KernelizedHopfieldNetwork):
            print("Training KHM kernel after document upload...")
            try:
                network.train_kernel(epochs=KHM_TRAIN_EPOCHS, lr=KHM_TRAIN_LR)
                # TODO: Save the trained kernel state if sqlite.py is enhanced
                print("KHM Kernel training complete.")
                # Re-save DB entries? Only if training modified embeddings/metadata (unlikely with current KHM setup)
                # save_network_to_db(network, db_manager, user_id)
            except Exception as train_e:
                print(f"ERROR during KHM kernel training: {str(train_e)}")
                # Decide if this should be a hard failure or just a warning
                # raise HTTPException(status_code=500, detail=f"Failed to train KHM kernel: {str(train_e)}")


        return {
            "status": "success",
            "document_id": document_id,
            "title": title,
            "memories_added": len(memories),
            "network_type": NETWORK_TYPE # Optional: return type used
        }

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    finally:
        # Clean up uploaded file
        if file_path_str and os.path.exists(file_path_str):
            try:
                os.remove(file_path_str)
            except OSError as remove_e:
                print(f"Error removing uploaded file {file_path_str}: {remove_e}")

# --- MODIFIED: Memory Query Endpoint ---
@app.post("/memories/query")
async def query_memories(query: MemoryQuery, request: Request):
    """Query the Hopfield network (Modern or KHM) for memories."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_id = user["id"]
        # Get user's network (appropriate type based on env var)
        # Load specific doc network if ID provided, otherwise user's general network
        network = get_user_network(user_id, query.document_id)

        total_memory_count_in_scope = db_manager.count_user_memories(user_id, query.document_id)
        if total_memory_count_in_scope == 0:
            print(f"No memories found in DB for user {user_id} (Doc: {query.document_id}). Returning empty.")
            return MemoryQueryResponse(results=[], gemini_response=None, use_gemini=query.use_gemini, retrieved_count=0, threshold_used=-2.0, similarity_scores=None) # Use -2 for context budget mode signal

        # Extract query and context (Keep as is)
        query_text = query.query_text
        chat_context_history = ""
        if "Previous messages:" in query_text and "New message:" in query_text:
             parts = query_text.split("New message:", 1)
             chat_context_history = parts[0].replace("Previous messages:", "").strip()
             query_text = parts[1].strip()

        # Embed query text
        query_embedding = text_processor.embed_text(query_text)

        candidate_memories: List[Memory] = [] # Memories to consider for context budget
        retrieval_mode_info = "" # For logging

        if isinstance(network, KernelizedHopfieldNetwork):
            print(f"Querying KHM network with beta={KHM_BETA}, threshold={query.similarity_threshold}")
            print(f"Querying KHM network (retrieving all ranked for context budget)")
            # KHM uses retrieve_with_kernel
            # Note: retrieve_with_kernel currently returns sorted memories, not scores directly.
            # We'd need to modify it or recalculate scores if needed for the response.
            ALL_THRESHOLD = -1.0
            candidate_memories = network.retrieve_with_kernel(
                query_embedding,
                beta=KHM_BETA,
                similarity_threshold=ALL_THRESHOLD, # Get all results sorted by score
                document_id=query.document_id
            )
            # TODO: Calculate/retrieve similarity scores if needed for KHM response
            # Example recalculation (potentially slow):
            # if retrieved_memories:
            #    mapped_query = network.feature_map(query_embedding.to(next(network.feature_map.parameters()).device))
            #    mapped_retrieved = network.feature_map(torch.stack([m.embeddings for m in retrieved_memories]).to(mapped_query.device))
            #    scores_tensor = F.cosine_similarity(mapped_query.unsqueeze(0), mapped_retrieved, dim=1)
            #    similarity_scores = scores_tensor.cpu().tolist()

        elif isinstance(network, ModernHopfieldNetwork):
            # For Modern, use the threshold first, then apply budget to those results
            print(f"Querying Modern Hopfield network with threshold={query.similarity_threshold}")
            retrieval_mode_info = f"Modern (Thresh: {query.similarity_threshold})"
            candidate_memories = network.retrieve(
                query_embedding,
                similarity_threshold=query.similarity_threshold,
                document_id=query.document_id
            )

        else:
            # raise HTTPException(status_code=500, detail="Internal server error: Unknown network type")
            print(f"[Warning] Unknown network type encountered: {type(network)}. Cannot perform retrieval.")
            candidate_memories = []

        print(f"{retrieval_mode_info} returned {len(candidate_memories)} candidate memories.")
        
        # --- Apply Context Budget ---
        context_limited_memories: List[Memory] = []
        current_context_length = 0
        if not candidate_memories:
            print("No candidate memories to process for context budget.")
        else:
            for memory in candidate_memories:
                mem_len = len(memory.original_text) # Use character length
                if (current_context_length + mem_len) <= CONTEXT_BUDGET:
                    context_limited_memories.append(memory)
                    current_context_length += mem_len
                else:
                    # Stop adding once budget is exceeded
                    print(f"Context budget ({CONTEXT_BUDGET} chars) reached after adding {len(context_limited_memories)} memories. Stopping.")
                    break # Stop iterating through candidates
            # Handle case where even the first memory exceeds the budget
            if not context_limited_memories and candidate_memories:
                 first_mem = candidate_memories[0]
                 print(f"First memory ({len(first_mem.original_text)} chars) already exceeds budget ({CONTEXT_BUDGET}). Including just the first one.")
                 context_limited_memories.append(first_mem)
                 current_context_length = len(first_mem.original_text)


        retrieved_memories = context_limited_memories # Final list after budgeting
        retrieved_count = len(retrieved_memories)
        
        # Format results (Keep as is, maybe add similarity score if available)
        results = []
        for i, memory in enumerate(retrieved_memories):
            result_item = {
                "memory_id": memory.memory_id,
                "text": memory.original_text,
                "name": memory.name,
                "section_title": memory.section_title,
                "timestamp": str(memory.timestamp) if memory.timestamp else None,
                "metadata": memory.metadata,
            }
            # Add score if calculated/available
            results.append(result_item)


        gemini_response_text = None
        if query.use_gemini:
            hopfield_context = ""
            # Apply max_gemini_context limit *on top of* the budgeted results
            memories_for_gemini = results[:query.max_gemini_context]

            if memories_for_gemini:
                print(f"Generating context for Gemini from {len(memories_for_gemini)} memories (budgeted & limited).")
                hopfield_context = format_context_from_memories(memories_for_gemini)
            else:
                print("No memories selected by context budget to send to Gemini.")
                hopfield_context = "No relevant context found in documents based on the query."

            full_context_for_gemini = ""
            if chat_context_history:
                full_context_for_gemini += f"CHAT HISTORY:\n{chat_context_history}\n\n"
            full_context_for_gemini += hopfield_context

            print("Querying Gemini...")
            gemini_response_text = query_gemini(query_text, full_context_for_gemini)
            print("Gemini response received.")

        # --- Return Response ---
        return MemoryQueryResponse(
            results=results,
            gemini_response=gemini_response_text,
            use_gemini=query.use_gemini,
            retrieved_count=retrieved_count, # Count after budgeting
            threshold_used=-2.0, # Use -2.0 (or similar) to indicate context budget mode
            similarity_scores=None # Scores are not directly used for filtering here
        )

    except Exception as e:
        print(f"Error processing memory query: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# --- MODIFIED: GET Memory Query Endpoint ---
@app.get("/memories/query")
async def query_memories_get(
    request: Request,
    query_text: str,
    document_id: Optional[str] = None,
    similarity_threshold: float = 0.75,
    use_gemini: bool = False,
    max_gemini_context: int = 5
):
    """GET endpoint to query the Hopfield network (Modern or KHM)."""
    # Create the Pydantic model from GET parameters
    query = MemoryQuery(
        query_text=query_text,
        document_id=document_id,
        similarity_threshold=similarity_threshold,
        use_gemini=use_gemini,
        max_gemini_context=max_gemini_context
    )
    # Call the POST endpoint's logic
    try:
        response_model = await query_memories(query=query, request=request)
        # Convert Pydantic model back to dict for JSON response
        return response_model.dict()
    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        # Handle other potential errors from the forwarded call
        print(f"GET Query failed with error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/network/stats")
async def get_network_stats(request: Request, document_id: Optional[str] = None):
    """Get statistics about a user's network (works for both types)."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = user["id"]
        # Get network (type doesn't strictly matter for counts from DB)
        network = get_user_network(user_id, document_id)

        # Get document info from the network object's state (if loaded/present)
        document_info = None
        if document_id and document_id in network.document_containers:
            container = network.document_containers[document_id]
            document_info = {
                "document_id": container.document_id,
                "title": container.title,
                "source_type": container.source_type,
                "timestamp": str(container.timestamp) if container.timestamp else None,
                # 'section_count': len(container.sections) # section count might be misleading if not fully loaded
            }
        # Get counts directly from the database for accuracy
        memory_count = db_manager.count_user_memories(user_id)
        document_count = db_manager.count_user_document_containers(user_id)

        return {
            "memory_count": memory_count,
            "document_count": document_count,
            "network_type": NETWORK_TYPE, # Include network type being used
            "document_info": document_info, # Info about a specific doc if requested and found
        }
    except Exception as e:
        print(f"Error getting network stats: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get network stats: {str(e)}")

# --- MODIFIED: File Query Endpoint ---
@app.post("/files/query")
async def query_files(query: FileQuery, request: Request):
    """Query the network for file content (works for both types, uses appropriate retrieve)."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_id = user["id"]
        network = get_user_network(user_id) # Get the correct network type
        query_embedding = text_processor.embed_text(query.query_text)

        retrieved_memories: List[Memory] = []
        similarity_scores: Optional[List[float]] = None

        # --- Conditional Retrieval ---
        if isinstance(network, KernelizedHopfieldNetwork):
            print(f"Querying KHM network for files with beta={KHM_BETA}, threshold={query.similarity_threshold}")
            # Use a low threshold for KHM file search initially? Or rely on its inherent separation?
            # Let's use the query's threshold, assuming kernel helps separate file content.
            retrieved_memories = network.retrieve_with_kernel(
                query_embedding,
                beta=KHM_BETA,
                similarity_threshold=query.similarity_threshold, # Use specified threshold
                document_id=None # Search all documents
            )
            # TODO: Calculate/retrieve scores if needed

        elif isinstance(network, ModernHopfieldNetwork):
            print(f"Querying Modern Hopfield network for files with threshold={query.similarity_threshold}")
            retrieved_memories = network.retrieve(
                query_embedding,
                similarity_threshold=query.similarity_threshold, # Use specified threshold
                document_id=None # Search all documents
            )
            # TODO: Get scores if ModernHopfieldNetwork returns them

        else:
            print(f"[Warning] Unknown network type encountered: {type(network)}. Cannot perform file query.")
            return {"results": []}


        # Filter and format results (Keep existing logic, maybe add score)
        file_results = []
        count = 0
        # We need to sort memories by score *after* retrieval if the retrieve method doesn't guarantee it,
        # especially if scores are recalculated or not part of the primary return.
        # For now, assume retrieve methods return reasonably sorted memories.
        # If scores were available:
        # memory_score_pairs = sorted(zip(retrieved_memories, similarity_scores), key=lambda x: x[1], reverse=True)
        # for memory, score in memory_score_pairs: ...

        for memory in retrieved_memories: # Iterate through potentially sorted memories
            if memory.metadata and 'file_type' in memory.metadata: # Check if it looks like a file memory
                if query.file_types and memory.metadata['file_type'] not in query.file_types:
                    continue

                result_item = {
                    "memory_id": memory.memory_id,
                    "file_name": memory.name,
                    "content": memory.original_text[:200] + ('...' if len(memory.original_text) > 200 else ''), # Truncate preview
                    "file_type": memory.metadata.get('file_type', 'unknown'),
                    "line_count": memory.metadata.get('line_count', 0),
                    "size_bytes": memory.metadata.get('size_bytes', 0),
                    "timestamp": str(memory.timestamp) if memory.timestamp else None,
                     # "relevance_score": round(score, 4) # If score was available
                }
                file_results.append(result_item)
                count += 1
                if count >= query.k: # Apply K limit *after* filtering and potentially sorting
                    break

        return {"results": file_results}

    except Exception as e:
        print(f"File query failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File query failed: {str(e)}")


# --- MODIFIED: GET File Query Endpoint ---
@app.get("/files/query")
async def query_files_get(
    request: Request,
    query_text: str,
    file_types: Optional[str] = None, # Comma-separated file types
    k: int = 5,
    similarity_threshold: float = 0.7 # Add threshold parameter
):
    """GET endpoint to query the network for file content."""
    file_types_list = file_types.split(',') if file_types else None
    # Create FileQuery model instance
    query = FileQuery(
        query_text=query_text,
        file_types=file_types_list,
        k=k,
        similarity_threshold=similarity_threshold # Pass threshold
    )
    # Call the POST endpoint's logic
    try:
        return await query_files(query=query, request=request)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"GET File Query failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File query failed: {str(e)}")

@app.get("/documents/{document_id}/content")
async def get_document_content(
    request: Request,
    document_id: str = FastApiPath(..., description="The ID of the document to retrieve content for")
):
    """Retrieve the reconstructed content of a specific document."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_id = user["id"]

    try:
        # 1. Get the document container metadata
        container_data = db_manager.get_document_container(document_id)
        if not container_data:
            raise HTTPException(status_code=404, detail="Document container not found")

        # 2. Verify user owns this document
        if container_data.get("user_id") != user_id:
             # Check if user_id is null or doesn't match
            is_dev_user_accessing_null = user_id == "dev_user_123" and container_data.get("user_id") is None
            if not is_dev_user_accessing_null:
                 raise HTTPException(status_code=403, detail="Not authorized to access this document")


        # 3. Get all memories for this document
        memories_data = db_manager.get_document_memories(document_id)

        if not memories_data:
            return {
                "document_id": document_id,
                "title": container_data.get("title", "Untitled"),
                "content": "", # Return empty content if no memories
                "source_type": container_data.get("source_type"),
                "metadata": container_data.get("metadata", {})
            }

        # 4. Sort memories based on section and chunk index from metadata
        def sort_key(memory):
            meta = memory.get("metadata", {})
            # Handle cases where indices might be missing, treat them as 0
            section_index = meta.get("section_index", 0)
            chunk_index = meta.get("chunk_index", 0)
             # Ensure consistent types for comparison if needed (e.g., convert to int if they might be strings)
            try:
                section_index = int(section_index)
            except (ValueError, TypeError):
                section_index = 0 # Fallback
            try:
                chunk_index = int(chunk_index)
            except (ValueError, TypeError):
                chunk_index = 0 # Fallback

            return (section_index, chunk_index)


        sorted_memories = sorted(memories_data, key=sort_key)

        # 5. Reconstruct the content by joining original_text
        # Add double newline between sections/chunks for better readability
        reconstructed_content = "\n\n".join([mem.get("original_text", "") for mem in sorted_memories])

        return {
            "document_id": document_id,
            "title": container_data.get("title", "Untitled"),
            "content": reconstructed_content,
            "source_type": container_data.get("source_type"),
             "metadata": container_data.get("metadata", {}) # Pass metadata too
        }

    except Exception as e:
        print(f"Error retrieving document content for {document_id}: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document content: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)