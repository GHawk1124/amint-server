from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
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
from sqlite import SQLiteManager, create_hopfield_network_from_db, save_network_to_db
from mistral import get_ocr_result
from dotenv import load_dotenv
import jwt
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware

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
    k: int = 5
    use_gemini: bool = False

class FileQuery(BaseModel):
    query_text: str
    file_types: Optional[List[str]] = None  # e.g., ["python", "javascript", "md"]
    k: int = 5

class DummyFileData(BaseModel):
    file_count: int = 5  # Number of dummy files to generate
    user_id: str  # User ID to add the files to

class MemoryQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    gemini_response: Optional[str] = None
    use_gemini: bool = False

# OAuth2 password bearer for token validation
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Helper function to get user from session
async def get_current_user(request: Request):
    """Get the current user from session"""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db_manager.get_user(user_id)

# Helper function to get a network for a user
def get_user_network(user_id: str, document_id: Optional[str] = None) -> ModernHopfieldNetwork:
    """Get or create a Hopfield network for a user"""
    try:
        # Try to load existing network from database
        network = create_hopfield_network_from_db(db_manager, document_id, user_id)
        return network
    except Exception as e:
        # If no network exists, create a new one
        print(f"Creating new network for user {user_id}: {str(e)}")
        network = ModernHopfieldNetwork(embedding_dim=384)  # Default embedding dim for MiniLM-L6-v2
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
        user_info = jwt.decode(id_token, options={"verify_signature": False})
        
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
            "token_expiry": (datetime.now() + timedelta(seconds=token_info.get("expires_in", 3600))).isoformat()
        }
        
        user_id = db_manager.create_user(user_data)
        
        # Save user ID in session
        request.session["user_id"] = user_id
        
        # Create initial empty network if user is new
        network = get_user_network(user_id)
        
        # Redirect to the frontend with success
        frontend_url = "http://localhost:1420" if not IS_PRODUCTION else "https://ghawk1124.github.io/amint"
        return RedirectResponse(f"{frontend_url}?auth=success")
    
    except Exception as e:
        print(f"Google OAuth error: {str(e)}")
        print(traceback.format_exc())
        # Redirect to the frontend with error
        frontend_url = "http://localhost:1420" if not IS_PRODUCTION else "https://ghawk1124.github.io/amint"
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
    """Upload a document file, process it with OCR if needed, and add to the Hopfield network"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file based on type
        if file_extension.lower() in ['.pdf', '.jpg', '.jpeg', '.png']:
            # Use Mistral OCR
            extracted_text = get_ocr_result(MISTRAL_API_KEY, str(file_path))
        else:
            # Assume it's a text file
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        
        # Create document container if not provided
        if not document_id:
            container = DocumentContainer(
                title=title,
                source_type="uploaded_file",
                metadata={"original_filename": file.filename}
            )
            # Create a properly structured dictionary
            container_data = {
                'document_id': container.document_id,
                'title': container.title,
                'timestamp': container.timestamp,
                'source_type': container.source_type,
                'metadata': container.metadata
            }
            # Explicitly create document container and get the ID
            document_id = db_manager.create_document_container(container_data, user_id)
            print(f"Created document container with ID: {document_id}")
        else:
            # Verify document container exists if ID was provided
            existing_container = db_manager.get_document_container(document_id)
            if not existing_container:
                raise HTTPException(status_code=404, detail="Document container not found")
            container = DocumentContainer(
                document_id=document_id,
                title=title
            )
        
        # Process the text and create memories
        container, memories = text_processor.process_document(
            title=title,
            text=extracted_text,
            chunk_size=3,
            max_chunk_length=1000,
            use_section_detection=True
        )
        
        # Make sure document_id is consistent
        if container.document_id != document_id:
            print(f"Warning: Document ID mismatch - updating from {container.document_id} to {document_id}")
            container.document_id = document_id
        
        # Update parent_id for all memories
        for memory in memories:
            memory.parent_id = document_id
        
        # Get user's network
        network = get_user_network(user_id, document_id)
        
        # Add memories to network
        network.batch_store(memories)
        
        # Save network to database
        save_stats = save_network_to_db(network, db_manager, user_id)
        print(f"Network save stats: {save_stats}")
        
        return {
            "status": "success",
            "document_id": document_id,
            "title": title,
            "memories_added": len(memories)
        }
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Clean up uploaded file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

@app.post("/memories/query")
async def query_memories(query: MemoryQuery, request: Request):
    """Query the Hopfield network for memories matching the input text"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        # Get user's network
        network = get_user_network(user_id, query.document_id)
        
        # Embed query text
        query_embedding = text_processor.embed_text(query.query_text)
        
        # Retrieve memories
        memories = network.retrieve(query_embedding, k=query.k, document_id=query.document_id)
        
        # Format response
        results = []
        for memory in memories:
            results.append({
                "memory_id": memory.memory_id,
                "text": memory.original_text,
                "name": memory.name,
                "section_title": memory.section_title,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                "metadata": memory.metadata
            })
        
        # Get Gemini response if use_gemini flag is set
        gemini_response = None
        if getattr(query, 'use_gemini', False) and results:
            # Format context from top memories
            context = format_context_from_memories(results[:5])  # Use top 5 memories
            # Query Gemini with user input and context
            gemini_response = query_gemini(query.query_text, context)
        
        return MemoryQueryResponse(
            results=results,
            gemini_response=gemini_response,
            use_gemini=getattr(query, 'use_gemini', False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/memories/query")
async def query_memories_get(
    request: Request,
    query_text: str,
    document_id: Optional[str] = None,
    k: int = 5,
    use_gemini: bool = False
):
    """GET endpoint to query the Hopfield network for memories matching the input text"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        # Get user's network
        network = get_user_network(user_id, document_id)
        
        # Embed query text
        query_embedding = text_processor.embed_text(query_text)
        
        # Retrieve memories
        memories = network.retrieve(query_embedding, k=k, document_id=document_id)
        
        # Format response
        results = []
        for memory in memories:
            results.append({
                "memory_id": memory.memory_id,
                "text": memory.original_text,
                "name": memory.name,
                "section_title": memory.section_title,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                "metadata": memory.metadata
            })
        
        # Get Gemini response if use_gemini flag is set
        gemini_response = None
        if use_gemini and results:
            # Format context from top memories (limited to top 5)
            top_memories = results[:5]
            context = format_context_from_memories(top_memories)
            # Query Gemini with user input and context
            gemini_response = query_gemini(query_text, context)
        
        return {
            "results": results,
            "gemini_response": gemini_response,
            "use_gemini": use_gemini
        }
    except Exception as e:
        import traceback
        print(f"Query failed with error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/network/stats")
async def get_network_stats(request: Request, document_id: Optional[str] = None):
    """Get statistics about a user's network"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        # Get user's network
        network = get_user_network(user_id, document_id)
        
        # Get document info if document_id is specified
        document_info = None
        if document_id and document_id in network.document_containers:
            container = network.document_containers[document_id]
            document_info = {
                "document_id": container.document_id,
                "title": container.title,
                "source_type": container.source_type,
                "timestamp": container.timestamp.isoformat() if container.timestamp else None,
                "section_count": len(container.sections)
            }
        
        # Count memories
        memory_count = len(network.memories)
        document_count = len(network.document_containers)
        
        return {
            "memory_count": memory_count,
            "document_count": document_count,
            "document_info": document_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get network stats: {str(e)}")

if not IS_PRODUCTION:
    @app.post("/dev/add-dummy-files")
    async def add_dummy_files(data: DummyFileData, request: Request):
        """Development endpoint to add dummy file data to a user's network for testing"""
        user = await get_current_user(request)
        if not user and data.user_id != "dev_user_123":  # Special case for dev user
            raise HTTPException(status_code=401, detail="Not authenticated")
            
        user_id = user["id"] if user else data.user_id
        print(f"[DEBUG] add_dummy_files called with user_id: {user_id}, file_count: {data.file_count}")
        
        try:
            # Get the user's network
            network = get_user_network(user_id)
            print(f"[DEBUG] User network retrieved successfully: {len(network.memories)} existing memories")
            
            # Create a document container
            document_container = DocumentContainer(
                title="Dummy Files Collection",
                source_type="code",
                metadata={"type": "test_data", "generated_at": datetime.now().isoformat()}
            )
            
            # Add the document container to the network
            print("[DEBUG] Adding document container to network...")
            network.add_document_container(document_container)
            
            # Sample code snippets with different file types
            file_samples = [
                {
                    "name": "app.py",
                    "content": """
    def main():
        print("Hello, world!")
        # Process items in a loop
        for i in range(10):
            print(f"Processing item {i}")
        return 0
    if __name__ == "__main__":
        main()
                    """,
                    "type": "python"
                },
                # ... [rest of the file samples remain the same]
            ]
            
            print(f"[DEBUG] Loaded {len(file_samples)} file samples")
            
            # Generate memories for each file
            memories = []
            print(f"[DEBUG] Starting to process {data.file_count} files...")
            
            # Use only the number of files requested (with cycling if necessary)
            for i in range(data.file_count):
                file_sample = file_samples[i % len(file_samples)]
                print(f"[DEBUG] Processing file {i+1}/{data.file_count}: {file_sample['name']}")
                
                try:
                    # Create memory for the file
                    # Embed the content using the text processor
                    print(f"[DEBUG] Embedding content for {file_sample['name']}...")
                    embedding = text_processor.embed_text(file_sample["content"])
                    print(f"[DEBUG] Embedding created successfully, shape: {embedding.shape}")
                    
                    memory = Memory(
                        name=file_sample["name"],
                        original_text=file_sample["content"],
                        embeddings=embedding,
                        parent_id=document_container.document_id,
                        metadata={
                            "file_type": file_sample["type"],
                            "line_count": len(file_sample["content"].split("\n")),
                            "size_bytes": len(file_sample["content"]),
                            "is_dummy": True
                        }
                    )
                    memories.append(memory)
                    print(f"[DEBUG] Memory object created for {file_sample['name']}")
                except Exception as file_error:
                    print(f"[DEBUG] Error processing file {file_sample['name']}: {str(file_error)}")
                    print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
                    raise  # Re-raise to be caught by the outer try-except
            
            # Add memories to the network
            print(f"[DEBUG] Adding {len(memories)} memories to network...")
            network.batch_store(memories)
            
            # Save to database with statistics
            print("[DEBUG] Saving network to database...")
            try:
                stats = save_network_to_db(network, db_manager, user_id)
                print(f"[DEBUG] Network saved successfully. Stats: {stats}")
            except Exception as db_error:
                print(f"[DEBUG] Database error: {str(db_error)}")
                print(f"[DEBUG] Database error traceback: {traceback.format_exc()}")
                stats = {"error": str(db_error)}
            
            return {
                "status": "success",
                "message": f"Processed {len(memories)} files",
                "document_id": document_container.document_id,
                "file_names": [m.name for m in memories],
                "stats": stats
            }
        except Exception as e:
            print(f"[DEBUG] Error in add_dummy_files: {str(e)}")
            print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Adding dummy files failed: {str(e)}")

@app.post("/files/query")
async def query_files(query: FileQuery, request: Request):
    """Query the Hopfield network for file content matching the input text"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        user_id = user["id"]
        # Get user's network
        network = get_user_network(user_id)
        
        # Embed query text
        query_embedding = text_processor.embed_text(query.query_text)
        
        # Retrieve memories
        memories = network.retrieve(query_embedding, k=query.k)
        
        # Filter by file types if specified
        file_results = []
        for memory in memories:
            # Check if this memory represents a file (has 'file_type' in metadata)
            if memory.metadata and 'file_type' in memory.metadata:
                # If file_types filter is applied and this file type is not in the list, skip it
                if query.file_types and memory.metadata['file_type'] not in query.file_types:
                    continue
                
                file_results.append({
                    "memory_id": memory.memory_id,
                    "file_name": memory.name,
                    "content": memory.original_text,
                    "file_type": memory.metadata.get('file_type', 'unknown'),
                    "line_count": memory.metadata.get('line_count', 0),
                    "size_bytes": memory.metadata.get('size_bytes', 0),
                    "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                    "relevance_score": round(float(torch.sigmoid(query_embedding @ memory.embeddings).item()), 4)
                })
        
        return {"results": file_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File query failed: {str(e)}")

@app.get("/files/query")
async def query_files_get(
    request: Request,
    query_text: str,
    file_types: Optional[str] = None,  # Comma-separated file types
    k: int = 5
):
    """GET endpoint to query the Hopfield network for file content matching the input text"""
    # Parse file_types from comma-separated string if provided
    file_types_list = file_types.split(',') if file_types else None
    
    # Create a FileQuery object from GET parameters
    query = FileQuery(query_text=query_text, file_types=file_types_list, k=k)
    
    # Reuse the POST endpoint logic
    return await query_files(query, request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)