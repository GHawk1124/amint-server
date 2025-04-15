# khm_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import traceback

# Import base classes from the original network file
# Adjust the import path if your file structure is different
try:
    from temporal_hopfield_network import Memory, DocumentContainer, MultiHeadRetrieval
except ImportError as e:
    print(f"Error importing base classes from temporal_hopfield_network: {e}")
    print("Please ensure temporal_hopfield_network.py is accessible.")
    # Handle import error (e.g., add to sys.path or exit)
    import sys
    sys.exit(1)


# --- KHM Implementation ---

class FeatureMap(nn.Module):
    """ Simple MLP feature map for KHM """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # Optional: Add LayerNorm for stability
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.norm(x) # Apply LayerNorm
        # Ensure output is normalized (unit length) for cosine similarity interpretation
        x = F.normalize(x, p=2, dim=-1)
        return x

def separation_loss(mapped_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculates the separation loss (U-Hop Stage I inspired).
    Aims to maximize distance (minimize cosine similarity) between *different* patterns.
    Args:
        mapped_embeddings: Embeddings after passing through the feature map (N x feature_dim), normalized.
    Returns:
        Scalar loss value.
    """
    if mapped_embeddings.shape[0] <= 1:
        return torch.tensor(0.0, device=mapped_embeddings.device) # No pairs to compare

    # Calculate pairwise cosine similarities in the feature space
    # S_ij = <phi(xi), phi(xj)>
    similarity_matrix = torch.matmul(mapped_embeddings, mapped_embeddings.T)

    # We want to minimize similarity for i != j.
    # Loss = sum_{i!=j} S_ij^2  (or other functions like sum |S_ij|, sum exp(S_ij))
    # Let's use mean squared similarity for off-diagonal elements.
    num_patterns = mapped_embeddings.shape[0]
    # Create a mask to exclude diagonal elements (similarity with self)
    mask = ~torch.eye(num_patterns, dtype=torch.bool, device=mapped_embeddings.device)

    # Calculate loss only on off-diagonal elements
    # Ensure mask has correct dimensions if similarity_matrix might be squeezed
    if similarity_matrix.ndim == 1 and mask.ndim == 2:
         mask = mask.squeeze() # Adjust mask if necessary
    elif similarity_matrix.ndim == 2 and mask.ndim == 1:
         # This case shouldn't happen with standard matmul, but handle defensively
         print("[Warning] Unexpected dimensions in separation_loss mask handling.")
         mask = mask.unsqueeze(0).expand_as(similarity_matrix)


    # Handle potential scalar output from matmul if only one embedding exists after mapping (shouldn't happen with check above)
    if similarity_matrix.numel() == 1:
         return torch.tensor(0.0, device=mapped_embeddings.device)


    # Check if mask shape matches similarity_matrix after potential adjustments
    if mask.shape != similarity_matrix.shape:
        print(f"[Warning] Mismatched shapes in separation_loss: similarity_matrix={similarity_matrix.shape}, mask={mask.shape}. Attempting to broadcast mask.")
        try:
            # Attempt broadcasting, might fail if incompatible
            mask = torch.broadcast_to(mask, similarity_matrix.shape)
        except RuntimeError as e:
            print(f"[Error] Broadcasting mask failed: {e}")
            return torch.tensor(torch.nan, device=mapped_embeddings.device) # Indicate error


    try:
        off_diagonal_similarities = similarity_matrix[mask]
        if off_diagonal_similarities.numel() == 0: # No off-diagonal elements (e.g., only 1 pattern)
             loss = torch.tensor(0.0, device=mapped_embeddings.device)
        else:
             loss = torch.mean(off_diagonal_similarities**2)

    except IndexError as e:
        print(f"[Error] IndexError during separation_loss calculation:")
        print(f"  Similarity Matrix Shape: {similarity_matrix.shape}")
        print(f"  Mask Shape: {mask.shape}")
        print(f"  Error: {e}")
        # Add more debug info if needed
        # print(f"  Similarity Matrix: {similarity_matrix}")
        # print(f"  Mask: {mask}")
        loss = torch.tensor(torch.nan, device=mapped_embeddings.device) # Return NaN or raise error

    return loss


class KernelizedHopfieldNetwork:
    """ KHM wrapper using a trainable feature map """
    def __init__(self, embedding_dim: int, feature_dim: int, hidden_dim: int, num_heads: int = 4):
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim # Dimension of the KHM feature space
        self.memories: List[Memory] = []
        self.document_containers: Dict[str, DocumentContainer] = {}

        # KHM Feature Map (Kernel)
        self.feature_map = FeatureMap(embedding_dim, hidden_dim, feature_dim)

        # Multi-Head Retrieval mechanism operating in the *feature* space
        # Ensure MultiHeadRetrieval is initialized correctly
        try:
            self.multi_head = MultiHeadRetrieval(feature_dim, num_heads) # Heads operate on feature_dim
        except Exception as e:
            print(f"Error initializing MultiHeadRetrieval in KHM: {e}")
            raise # Re-raise the error

    def get_all_embeddings(self) -> Optional[torch.Tensor]:
        """ Returns a stacked tensor of all memory embeddings. """
        if not self.memories:
            return None
        # Ensure all embeddings are tensors before stacking
        embeddings_list = []
        for m in self.memories:
            if isinstance(m.embeddings, torch.Tensor):
                embeddings_list.append(m.embeddings)
            else:
                # Handle potential non-tensor data if loading from unexpected source
                print(f"[Warning] Memory {m.memory_id} has non-tensor embedding type: {type(m.embeddings)}. Attempting conversion.")
                try:
                    embeddings_list.append(torch.tensor(m.embeddings, dtype=torch.float32))
                except Exception as conv_e:
                    print(f"[Error] Failed to convert embedding for memory {m.memory_id} to tensor: {conv_e}")
                    return None # Cannot proceed if conversion fails
        if not embeddings_list:
             return None
        return torch.stack(embeddings_list)


    def train_kernel(self, epochs: int = 50, lr: float = 0.001):
        """ Train the feature map using separation loss (U-Hop Stage I inspired). """
        print(f"\n--- Training KHM Kernel (Feature Map) ---")
        print(f"Epochs: {epochs}, LR: {lr}")
        embeddings = self.get_all_embeddings()
        if embeddings is None or embeddings.shape[0] <= 1:
            print("Not enough memories (<2) to train the kernel. Skipping.")
            return

        self.feature_map.train() # Set model to training mode
        optimizer = optim.Adam(self.feature_map.parameters(), lr=lr)

        # Move model and data to the same device (e.g., GPU if available)
        device = next(self.feature_map.parameters()).device
        embeddings = embeddings.to(device)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Map embeddings to feature space
            mapped_embeddings = self.feature_map(embeddings)

            # Calculate separation loss
            loss = separation_loss(mapped_embeddings)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[Error] NaN loss encountered during kernel training at epoch {epoch+1}. Stopping.")
                break

            if loss.item() == 0.0 and embeddings.shape[0] > 1:
                # Check if loss is exactly zero (might indicate perfect separation or issue)
                 is_constant = torch.allclose(mapped_embeddings[0], mapped_embeddings)
                 if not is_constant: # If not all embeddings mapped to the same point
                    print(f"Epoch {epoch+1}/{epochs} - Loss is zero, stopping early (potential perfect separation or convergence).")
                    break
                 else: # If all mapped embeddings are the same, loss is zero but separation failed
                    print(f"Epoch {epoch+1}/{epochs} - Loss is zero, but all embeddings mapped to the same point. Training may not be effective.")
                    # Optionally continue training or add checks

            # Backpropagate and update weights
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as grad_e:
                 print(f"[Error] Gradient calculation/optimizer step failed at epoch {epoch+1}: {grad_e}")
                 print(traceback.format_exc())
                 break # Stop training if backward/step fails


            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Separation Loss: {loss.item():.6f}")

        self.feature_map.eval() # Set model back to evaluation mode
        print("--- Kernel Training Complete ---")


    def store(self, memory: Memory):
        """ Store a memory. Kernel should be retrained if many memories are added. """
        # Basic storage, assumes kernel is trained separately or afterwards
        self.memories.append(memory)
        if memory.parent_id and memory.parent_id in self.document_containers:
            self.document_containers[memory.parent_id].add_section(memory.memory_id)

    def batch_store(self, memories: List[Memory]):
         for memory in memories:
             self.store(memory)

    def add_document_container(self, container: DocumentContainer):
        self.document_containers[container.document_id] = container

    def get_document_memories(self, document_id: str) -> List[Memory]:
        """ Get memories belonging to a specific document. """
        if document_id not in self.document_containers:
            return []
        container = self.document_containers[document_id]
        # Need to efficiently map memory_ids in container.sections back to Memory objects
        doc_memory_ids = set(container.sections)
        return [mem for mem in self.memories if mem.memory_id in doc_memory_ids]

    def retrieve_with_kernel(self,
                             query_embedding: torch.Tensor,
                             beta: float = 8.0,
                             similarity_threshold: float = 0.7,
                             document_id: Optional[str] = None
                             ) -> List[Memory]:
        """ Retrieve memories using the trained kernel (feature map). """
        if not self.memories:
            return []

        self.feature_map.eval() # Ensure feature map is in eval mode
        
        try:

            target_memories = self.get_document_memories(document_id) if document_id else self.memories
            if not target_memories:
                return []

            # Ensure query_embedding is a tensor
            if not isinstance(query_embedding, torch.Tensor):
                print(f"[Warning] Query embedding is not a tensor ({type(query_embedding)}). Attempting conversion.")
                try:
                    query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
                except Exception as q_conv_e:
                    print(f"[Error] Failed to convert query embedding to tensor: {q_conv_e}")
                    return []

            memory_embeddings_orig = torch.stack([mem.embeddings for mem in target_memories])

            # Move data to the same device as the model
            device = next(self.feature_map.parameters()).device
            query_embedding = query_embedding.to(device)
            memory_embeddings_orig = memory_embeddings_orig.to(device)


            # --- KHM Step: Map query and memories to feature space ---
            with torch.no_grad(): # No need to track gradients during retrieval
                mapped_query = self.feature_map(query_embedding)
                mapped_memories = self.feature_map(memory_embeddings_orig)
                similarity_scores = F.cosine_similarity(mapped_query.unsqueeze(0), mapped_memories, dim=1) # Shape (N)
                indices = torch.where(similarity_scores >= similarity_threshold)[0]
                if indices.numel() == 0:
                    return []
                # Get the original memories and their scores
                indices_cpu = indices.cpu()
                retrieved_memories_orig = [target_memories[idx.item()] for idx in indices_cpu]
                retrieved_scores = similarity_scores[indices] # Use the direct scores

                # Sort results by similarity score (descending)
                retrieved_scores_cpu = retrieved_scores.cpu()
                sorted_indices_local = torch.argsort(retrieved_scores_cpu, descending=True)
                sorted_memories = [retrieved_memories_orig[i] for i in sorted_indices_local]

                # Optional: log scores
                print(f"[INFO] KHM Direct Retrieval: Found {len(sorted_memories)} memories above threshold {similarity_threshold}.")
                print(f"[INFO] Scores (Min/Max/Avg): {retrieved_scores_cpu.min():.4f} / {retrieved_scores_cpu.max():.4f} / {retrieved_scores_cpu.mean():.4f}")
                
                return sorted_memories

            # --- End KHM Step ---

        except Exception as e:
            print(f"Error during KHM retrieval: {e}")
            print(traceback.format_exc())
            return []