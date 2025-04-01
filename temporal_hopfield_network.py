import numpy as np
import uuid
import re
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

class Memory:
    """Class to store memory entries with metadata and embeddings"""
    def __init__(
        self, 
        name: str, 
        original_text: str, 
        embeddings: Union[np.ndarray, torch.Tensor],
        memory_id: str = None,
        timestamp: datetime = None,
        parent_id: str = None,
        section_title: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name  # File name, chat name, etc.
        self.memory_id = memory_id if memory_id else str(uuid.uuid4())
        self.timestamp = timestamp if timestamp else datetime.now()
        self.original_text = original_text
        
        # New fields for hierarchical storage
        self.parent_id = parent_id  # ID of parent document/container
        self.section_title = section_title  # Title of this section if applicable
        self.metadata = metadata if metadata else {}  # For storing additional info like page numbers, chapters, etc.
        
        # Convert embeddings to torch tensor if they are numpy arrays
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        else:
            self.embeddings = embeddings

class DocumentContainer:
    """Container for organizing memories hierarchically, useful for textbooks, long documents"""
    def __init__(
        self,
        title: str,
        document_id: str = None,
        timestamp: datetime = None,
        source_type: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.title = title
        self.document_id = document_id if document_id else str(uuid.uuid4())
        self.timestamp = timestamp if timestamp else datetime.now()
        self.source_type = source_type  # e.g., "textbook", "article", "transcript"
        self.metadata = metadata if metadata else {}
        self.sections = []  # List of section IDs (memory_ids) belonging to this document
        
    def add_section(self, memory_id: str) -> None:
        """Add a section (memory) to this document container"""
        if memory_id not in self.sections:
            self.sections.append(memory_id)
            
    def remove_section(self, memory_id: str) -> bool:
        """Remove a section from this document container"""
        if memory_id in self.sections:
            self.sections.remove(memory_id)
            return True
        return False

class ModernHopfieldNetwork:
    """Implementation of a Modern Hopfield Network for continuous patterns"""
    def __init__(self, embedding_dim: int, beta: float = 8.0):
        """
        Initialize a Modern Hopfield Network.
        
        Args:
            embedding_dim: Dimensionality of the embeddings
            beta: Temperature parameter for the softmax function (controls pattern separation)
        """
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.memories = []
        self.document_containers = {}  # Document ID -> DocumentContainer mapping
        
    def store(self, memory: Memory) -> None:
        """
        Store a new memory in the network (one-shot learning).
        
        Args:
            memory: Memory object to be stored
        """
        self.memories.append(memory)
        
        # If this memory belongs to a document container, update the container
        if memory.parent_id and memory.parent_id in self.document_containers:
            self.document_containers[memory.parent_id].add_section(memory.memory_id)
    
    def batch_store(self, memories: List[Memory]) -> None:
        """
        Store multiple memories at once.
        
        Args:
            memories: List of Memory objects to be stored
        """
        for memory in memories:
            self.store(memory)
    
    def add_document_container(self, container: DocumentContainer) -> None:
        """
        Add a document container to organize memories hierarchically.
        
        Args:
            container: DocumentContainer object to add
        """
        self.document_containers[container.document_id] = container
        
    def get_document_memories(self, document_id: str) -> List[Memory]:
        """
        Get all memories belonging to a specific document.
        
        Args:
            document_id: ID of the document to retrieve memories for
            
        Returns:
            List of memories belonging to the document
        """
        if document_id not in self.document_containers:
            return []
            
        container = self.document_containers[document_id]
        return [memory for memory in self.memories if memory.memory_id in container.sections]
    
    def retrieve(self, query_embedding: torch.Tensor, k: int = 1, document_id: Optional[str] = None) -> List[Memory]:
        """
        Retrieve the k most similar memories to the query embedding.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of memories to retrieve
            document_id: Optional document ID to restrict search to
            
        Returns:
            List of the k most similar memories
        """
        if not self.memories:
            return []
        
        # If document_id is provided, restrict search to memories from that document
        target_memories = self.get_document_memories(document_id) if document_id else self.memories
        
        if not target_memories:
            return []
        
        # Stack all memory embeddings
        memory_embeddings = torch.stack([memory.embeddings for memory in target_memories])
        
        # Compute similarity between query and all memories
        similarity = torch.matmul(query_embedding, memory_embeddings.T)
        
        # Apply softmax with temperature beta (core of modern Hopfield update rule)
        attention = F.softmax(self.beta * similarity, dim=0)
        
        # Get the indices of the k memories with highest attention
        _, indices = torch.topk(attention, min(k, len(target_memories)))
        
        # Return the corresponding memories
        return [target_memories[idx.item()] for idx in indices]
    
    def update(self, memory_id: str, new_embedding: torch.Tensor, new_text: Optional[str] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            new_embedding: New embedding for the memory
            new_text: New original text (optional)
            
        Returns:
            Boolean indicating success
        """
        for i, memory in enumerate(self.memories):
            if memory.memory_id == memory_id:
                memory.embeddings = new_embedding
                if new_text:
                    memory.original_text = new_text
                memory.timestamp = datetime.now()  # Update timestamp
                return True
        return False
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory from the network.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            Boolean indicating success
        """
        # Remove from any document containers first
        for container in self.document_containers.values():
            container.remove_section(memory_id)
            
        for i, memory in enumerate(self.memories):
            if memory.memory_id == memory_id:
                self.memories.pop(i)
                return True
        return False
    
    def delete_document(self, document_id: str, delete_memories: bool = True) -> bool:
        """
        Delete a document container and optionally its memories.
        
        Args:
            document_id: ID of the document to delete
            delete_memories: Whether to also delete the memories belonging to the document
            
        Returns:
            Boolean indicating success
        """
        if document_id not in self.document_containers:
            return False
            
        # Get the memories belonging to this document
        if delete_memories:
            memory_ids = self.document_containers[document_id].sections.copy()
            for memory_id in memory_ids:
                self.delete(memory_id)
                
        # Remove the document container
        del self.document_containers[document_id]
        return True
    
    def clear(self) -> None:
        """Clear all memories and document containers from the network."""
        self.memories = []
        self.document_containers = {}
        
    def save(self, filepath: str) -> None:
        """
        Save the network to a file.
        
        Args:
            filepath: Path to save the network
        """
        torch.save({
            'embedding_dim': self.embedding_dim,
            'beta': self.beta,
            'memories': self.memories,
            'document_containers': self.document_containers
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModernHopfieldNetwork':
        """
        Load a network from a file.
        
        Args:
            filepath: Path to load the network from
            
        Returns:
            Loaded ModernHopfieldNetwork
        """
        checkpoint = torch.load(filepath)
        network = cls(
            embedding_dim=checkpoint['embedding_dim'],
            beta=checkpoint['beta']
        )
        network.memories = checkpoint['memories']
        network.document_containers = checkpoint.get('document_containers', {})
        return network

class TextProcessor:
    """Handles text chunking and embedding generation"""
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize a TextProcessor.
        
        Args:
            embedding_model_name: Name of the pre-trained SentenceTransformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def chunk_text(self, text: str, chunk_size: int = 3, max_chunk_length: int = 1000) -> List[str]:
        """
        Split text into chunks of sentences.
        
        Args:
            text: Text to split
            chunk_size: Number of sentences per chunk
            max_chunk_length: Maximum length of a chunk in characters
            
        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed max_chunk_length and we already have sentences,
            # complete the current chunk and start a new one
            if current_length + len(sentence) > max_chunk_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            # If we've reached the desired chunk_size, complete the chunk
            if len(current_chunk) >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add any remaining text as the final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def identify_sections(self, text: str, section_patterns: List[str] = None) -> List[Tuple[str, str]]:
        """
        Split text into sections based on headers or patterns.
        
        Args:
            text: Text to split into sections
            section_patterns: Regex patterns to identify section headers. If None, uses default patterns.
            
        Returns:
            List of (section_title, section_content) tuples
        """
        if section_patterns is None:
            # Default patterns for identifying headers in academic/textbook content
            section_patterns = [
                r"(?:^|\n)#+\s+(.+?)(?:\n|$)",  # Markdown headers
                r"(?:^|\n)Chapter\s+\d+[:\.\s]+(.+?)(?:\n|$)",  # Chapter headers
                r"(?:^|\n)Section\s+\d+(?:\.\d+)*[:\.\s]+(.+?)(?:\n|$)",  # Section headers
                r"(?:^|\n)(?:\d+(?:\.\d+)*)[:\.\s]+(.+?)(?:\n|$)",  # Numbered sections (e.g., "1.2.3 Section Title")
                r"(?:^|\n)([A-Z][A-Za-z\s]+)(?:\n|$)"  # All caps or title case potential headers
            ]
            
        # Combine patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
        
        # Find all matches
        matches = list(re.finditer(combined_pattern, text))
        
        if not matches:
            # If no sections found, return the entire text as one section
            return [("Main Content", text)]
            
        sections = []
        for i, match in enumerate(matches):
            # Extract the section title from the match
            title = None
            for group_idx in range(1, len(match.groups()) + 1):
                if match.group(group_idx):
                    title_match = re.search(r"(.+?)(?:\n|$)", match.group(group_idx))
                    if title_match:
                        title = title_match.group(1).strip()
                        break
            
            if not title:
                title = f"Section {i+1}"
                
            # Section content goes from this match to the next match (or end of text)
            start_pos = match.start()
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            content = text[start_pos:end_pos].strip()
            
            sections.append((title, content))
            
        return sections
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Create an embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a torch tensor
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding
    
    def embed_chunks(self, chunks: List[str]) -> List[torch.Tensor]:
        """
        Create embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embeddings as torch tensors
        """
        embeddings = [self.embed_text(chunk) for chunk in chunks]
        return embeddings
    
    def process_text(self, name: str, text: str, chunk_size: int = 3, max_chunk_length: int = 1000) -> List[Memory]:
        """
        Process a text into chunks and create Memory objects.
        
        Args:
            name: Name for the memories
            text: Text to process
            chunk_size: Number of sentences per chunk
            max_chunk_length: Maximum length of a chunk in characters
            
        Returns:
            List of Memory objects
        """
        chunks = self.chunk_text(text, chunk_size, max_chunk_length)
        embeddings = self.embed_chunks(chunks)
        
        memories = []
        for chunk, embedding in zip(chunks, embeddings):
            memory = Memory(
                name=name,
                original_text=chunk,
                embeddings=embedding
            )
            memories.append(memory)
            
        return memories
        
    def process_document(
        self, 
        title: str, 
        text: str, 
        chunk_size: int = 3, 
        max_chunk_length: int = 1000,
        use_section_detection: bool = True,
        section_patterns: List[str] = None,
        source_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[DocumentContainer, List[Memory]]:
        """
        Process a full document including section detection and create Memory objects with hierarchy.
        
        Args:
            title: Title of the document
            text: Document text to process
            chunk_size: Number of sentences per chunk
            max_chunk_length: Maximum length of a chunk in characters
            use_section_detection: Whether to try to detect sections in the document
            section_patterns: Custom regex patterns to identify section headers
            source_type: Type of document (e.g., "textbook", "article")
            metadata: Additional metadata for the document
            
        Returns:
            Tuple of (DocumentContainer, List[Memory])
        """
        # Create a document container
        document = DocumentContainer(
            title=title,
            source_type=source_type,
            metadata=metadata
        )
        
        all_memories = []
        
        if use_section_detection:
            # Try to detect sections
            sections = self.identify_sections(text, section_patterns)
            
            for section_title, section_content in sections:
                # Process each section
                chunks = self.chunk_text(section_content, chunk_size, max_chunk_length)
                embeddings = self.embed_chunks(chunks)
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_title = f"{section_title} (Part {i+1})" if len(chunks) > 1 else section_title
                    
                    memory = Memory(
                        name=title,
                        original_text=chunk,
                        embeddings=embedding,
                        parent_id=document.document_id,
                        section_title=section_title,
                        metadata={"chunk_index": i, "section_index": sections.index((section_title, section_content))}
                    )
                    
                    all_memories.append(memory)
                    document.add_section(memory.memory_id)
        else:
            # Process without section detection
            chunks = self.chunk_text(text, chunk_size, max_chunk_length)
            embeddings = self.embed_chunks(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                memory = Memory(
                    name=title,
                    original_text=chunk,
                    embeddings=embedding,
                    parent_id=document.document_id,
                    metadata={"chunk_index": i}
                )
                
                all_memories.append(memory)
                document.add_section(memory.memory_id)
                
        return document, all_memories

# Example usage
def example_usage():
    print("Downloading tokenizer if not already available...")
    # Download tokenizer if not already available
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("Tokenizer downloaded.")

    # Initialize the text processor
    processor = TextProcessor()
    
    # Initialize the Hopfield Network
    embedding_dim = processor.embedding_model.get_sentence_embedding_dimension()
    hopfield = ModernHopfieldNetwork(embedding_dim)
    
    # Process and store some text
    text1 = "Modern Hopfield Networks are a powerful tool for memory storage and retrieval. They extend traditional Hopfield Networks to continuous patterns. This allows for much higher storage capacity."
    memories1 = processor.process_text("Hopfield_Info", text1)
    hopfield.batch_store(memories1)
    
    text2 = "Python is a programming language. It is widely used for data science and machine learning. Its simple syntax makes it easy to learn."
    memories2 = processor.process_text("Python_Info", text2)
    hopfield.batch_store(memories2)
    
    # Process a document with sections
    textbook_sample = """
    # Chapter 1: Introduction to Machine Learning
    
    Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.
    The goal is to progressively improve performance on a specific task without being explicitly programmed.
    Machine learning algorithms build a model based on sample data in order to make predictions or decisions.
    
    ## 1.1 Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    Supervised learning algorithms build a model based on labeled training data.
    Unsupervised learning algorithms find patterns in unlabeled data.
    
    ## 1.2 Applications
    
    Machine learning has many applications including recommendation systems, natural language processing, and computer vision.
    Recommendation systems suggest products or content based on user behavior and preferences.
    Natural language processing enables computers to understand and generate human language.
    """
    
    document, document_memories = processor.process_document(
        title="Machine Learning Textbook",
        text=textbook_sample,
        chunk_size=2,
        use_section_detection=True,
        source_type="textbook"
    )
    
    # Add the document container to the network
    hopfield.add_document_container(document)
    
    # Store all the document memories
    hopfield.batch_store(document_memories)
    
    # Retrieve memories based on a query
    query = "Tell me about neural networks for memory storage"
    query_embedding = processor.embed_text(query)
    
    # Retrieve from all memories
    retrieved_memories = hopfield.retrieve(query_embedding, k=2)
    
    print("Query:", query)
    print("Retrieved memories:")
    for memory in retrieved_memories:
        print(f"- Name: {memory.name}")
        print(f"  Text: {memory.original_text}")
        print(f"  Time: {memory.timestamp}")
        if memory.section_title:
            print(f"  Section: {memory.section_title}")
        print()
    
    # Retrieve only from the textbook document
    print("\nRetrieving only from the textbook document:")
    document_memories = hopfield.retrieve(query_embedding, k=2, document_id=document.document_id)
    
    for memory in document_memories:
        print(f"- Section: {memory.section_title}")
        print(f"  Text: {memory.original_text}")
        print()

if __name__ == "__main__":
    example_usage()