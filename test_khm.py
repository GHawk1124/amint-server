# test_advanced_hopfield.py
import pytest
import torch
# No longer need nn, optim here as they are in khm_network.py
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
import math
# traceback import might still be useful here if test setup fails
import traceback

# --- Assuming access to your existing modules ---
try:
    # Import base classes needed for test setup and typing
    from temporal_hopfield_network import Memory, DocumentContainer, TextProcessor
    # Import the NEW KHM class from its dedicated file
    from khm_network import KernelizedHopfieldNetwork # <--- MODIFIED IMPORT
except ImportError as e:
    print(f"Import Error: {e}. Ensure project modules are accessible.")
    import sys
    sys.exit(1)

# --- Configuration --- (Keep configuration here)
DOCUMENTS = {
    "doc1": {
        "title": "Solar System Basics",
        "text": """The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.
Mercury is the closest planet to the Sun. Venus is the second planet.
Earth is the third planet and the only one known to support life. Mars is the fourth planet, often called the Red Planet.
Jupiter is the largest planet, a gas giant. Saturn is known for its prominent rings.
Uranus and Neptune are ice giants in the outer Solar System.""",
        "source_type": "article"
    },
    "doc2": {
        "title": "Python Programming",
        "text": """Python is a high-level, interpreted programming language. It emphasizes code readability.
Guido van Rossum began working on Python in the late 1980s.
Its standard library is large and comprehensive. Python supports multiple programming paradigms, including structured, object-oriented and functional programming.
It is often used for web development, data science, artificial intelligence, and scripting.""",
        "source_type": "reference"
    },
    "doc3": {
        "title": "Apollo 11 Moon Landing",
        "text": """Apollo 11 was the American spaceflight that first landed humans on the Moon.
Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.
Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later. Aldrin joined him about 20 minutes later.
They spent about two and a quarter hours together outside the spacecraft, collecting lunar material. Command module pilot Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface.""",
        "source_type": "historical_summary"
    },
    "doc4": {
        "title": "Photosynthesis Overview",
        "text": """Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. This chemical energy is later released to fuel the organisms' activities.
This process occurs in chloroplasts, using chlorophyll, the pigment that gives plants their green color. Water and carbon dioxide are used as reactants.
Oxygen is released as a byproduct. Glucose (a sugar) is produced to store energy.""",
        "source_type": "biology_concept"
    }
}

QUERIES_GROUND_TRUTH = {
    "What is the Solar System?": {"doc1": {0}},
    "Which planets are gas giants?": {"doc1": {2}},
    "Tell me about Python's history": {"doc2": {1}},
    "What is Python used for?": {"doc2": {2}},
    "Does Earth have rings?": {"doc1": {1, 3}},
    "Who landed on the Moon in 1969?": {"doc3": {0}},
    "What did the Apollo 11 astronauts collect?": {"doc3": {2}},
    "Who was the command module pilot for Apollo 11?": {"doc3": {2}},
    "What is photosynthesis?": {"doc4": {0}},
    "What are the inputs for photosynthesis?": {"doc4": {1}},
    "What does photosynthesis release?": {"doc4": {2}},
    "Where does photosynthesis happen in plants?": {"doc4": {1}}
}

CHUNK_SIZE = 2
MAX_CHUNK_LENGTH = 1000
BETA_VALUES = [8.0, 16.0, 32.0]
SIMILARITY_THRESHOLD = 0.7
KERNEL_TRAINING_EPOCHS = 50
KERNEL_LEARNING_RATE = 0.001
FEATURE_MAP_HIDDEN_DIM = 512

# --- KHM Implementation ---
# REMOVED: FeatureMap class definition
# REMOVED: separation_loss function definition
# REMOVED: KernelizedHopfieldNetwork class definition
# (These are now in khm_network.py)

# --- Pytest Fixtures --- (Keep fixtures here)

@pytest.fixture(scope="module")
def text_processor() -> TextProcessor:
    """Initialize the TextProcessor once per module."""
    print("\nInitializing Text Processor...")
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        import nltk
        nltk.download('punkt', quiet=True)
    return TextProcessor()

@pytest.fixture(scope="module")
def setup_advanced_network(text_processor: TextProcessor) -> Tuple[KernelizedHopfieldNetwork, Dict[str, Dict[int, str]]]:
    """
    Initialize the KHM network, process documents, train the kernel, and store memories.
    Returns the trained network and the memory_id_map.
    """
    print("\nSetting up Kernelized Hopfield Network for testing...")
    try:
        embedding_dim = text_processor.embedding_model.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"Error getting embedding dimension: {e}. Defaulting to 384.")
        embedding_dim = 384

    feature_dim = embedding_dim

    # Use the imported KernelizedHopfieldNetwork class
    network = KernelizedHopfieldNetwork(
        embedding_dim=embedding_dim,
        feature_dim=feature_dim,
        hidden_dim=FEATURE_MAP_HIDDEN_DIM
    )

    memory_id_map: Dict[str, Dict[int, str]] = {}
    all_memories_to_store = []

    for doc_id, doc_data in DOCUMENTS.items():
        print(f"Processing document: {doc_id} - {doc_data['title']}")
        document_container, memories = text_processor.process_document(
            title=doc_data['title'],
            text=doc_data['text'],
            chunk_size=CHUNK_SIZE,
            max_chunk_length=MAX_CHUNK_LENGTH,
            use_section_detection=False,
            source_type=doc_data['source_type'],
            metadata={"original_doc_id": doc_id}
        )
        network.add_document_container(document_container)
        all_memories_to_store.extend(memories)

        memory_id_map[doc_id] = {}
        for i, mem in enumerate(memories):
            chunk_index = mem.metadata.get("chunk_index", i)
            memory_id_map[doc_id][chunk_index] = mem.memory_id

    network.batch_store(all_memories_to_store)
    print(f"Stored {len(network.memories)} memories.")

    # Train the Kernel
    network.train_kernel(epochs=KERNEL_TRAINING_EPOCHS, lr=KERNEL_LEARNING_RATE)

    print("Setup complete. KHM Network is ready and kernel trained.")
    return network, memory_id_map

# --- Helper Function --- (Keep helper functions here)
def calculate_metrics(retrieved_ids: Set[str], relevant_ids: Set[str]) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1 Score."""
    tp = len(retrieved_ids.intersection(relevant_ids))
    fp = len(retrieved_ids.difference(relevant_ids))
    fn = len(relevant_ids.difference(retrieved_ids))

    if not relevant_ids:
        precision = 1.0 if not retrieved_ids else 0.0
        recall = 1.0
        f1 = 1.0 if not retrieved_ids else 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": float(tp), "fp": float(fp), "fn": float(fn)
    }

# --- Test Function --- (Keep test functions here)
@pytest.mark.parametrize("query", QUERIES_GROUND_TRUTH.keys())
@pytest.mark.parametrize("beta", BETA_VALUES)
def test_advanced_retrieval_metrics(
    query: str,
    beta: float,
    setup_advanced_network: Tuple[KernelizedHopfieldNetwork, Dict[str, Dict[int, str]]],
    text_processor: TextProcessor
):
    """ Tests KHM retrieval performance for a given query and beta value. """
    network, memory_id_map = setup_advanced_network
    print(f"\n--- Testing KHM Query: '{query}' | Beta: {beta} | Threshold: {SIMILARITY_THRESHOLD} ---")

    # 1. Get Ground Truth Memory IDs
    relevant_chunk_indices_by_doc = QUERIES_GROUND_TRUTH.get(query, {})
    ground_truth_memory_ids: Set[str] = set()
    for doc_id, relevant_indices in relevant_chunk_indices_by_doc.items():
        if doc_id in memory_id_map:
             doc_map = memory_id_map[doc_id]
             for chunk_idx in relevant_indices:
                 if chunk_idx in doc_map:
                     ground_truth_memory_ids.add(doc_map[chunk_idx])
                 else:
                     print(f"[Warning] Ground truth chunk index {chunk_idx} not found in memory map for doc {doc_id}")
        else:
             print(f"[Warning] Ground truth doc_id {doc_id} not found in memory map")

    # 2. Perform KHM Retrieval
    query_embedding = text_processor.embed_text(query)
    retrieved_memories = network.retrieve_with_kernel(
        query_embedding,
        beta=beta,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    retrieved_memory_ids = {mem.memory_id for mem in retrieved_memories}

    # 3. Calculate Metrics
    metrics = calculate_metrics(retrieved_memory_ids, ground_truth_memory_ids)

    # 4. Output / Assertions
    print(f"  Relevant IDs (Expected): {ground_truth_memory_ids or '{}'}")
    print(f"  Retrieved IDs (Actual):  {retrieved_memory_ids or '{}'}")
    print(f"  Metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f} (TP={int(metrics['tp'])}, FP={int(metrics['fp'])}, FN={int(metrics['fn'])})")

# --- Optional: Test for Overall Average Performance --- (Keep average test here)
def test_advanced_average_performance(
    setup_advanced_network: Tuple[KernelizedHopfieldNetwork, Dict[str, Dict[int, str]]],
    text_processor: TextProcessor
):
    """ Calculates average metrics across all queries for each beta value. """
    network, memory_id_map = setup_advanced_network
    print("\n--- Testing KHM Average Performance Across Beta Values ---")
    overall_results: Dict[float, Dict[str, float]] = {}

    for beta in BETA_VALUES:
        threshold_metrics_list = []
        for query in QUERIES_GROUND_TRUTH.keys():
            # --- Repeat retrieval and metric calculation ---
            relevant_chunk_indices_by_doc = QUERIES_GROUND_TRUTH.get(query, {})
            ground_truth_memory_ids: Set[str] = set()
            for doc_id, relevant_indices in relevant_chunk_indices_by_doc.items():
                if doc_id in memory_id_map:
                    doc_map = memory_id_map[doc_id]
                    for chunk_idx in relevant_indices:
                        if chunk_idx in doc_map:
                            ground_truth_memory_ids.add(doc_map[chunk_idx])

            query_embedding = text_processor.embed_text(query)
            retrieved_memories = network.retrieve_with_kernel(
                query_embedding,
                beta=beta,
                similarity_threshold=SIMILARITY_THRESHOLD
            )
            retrieved_memory_ids = {mem.memory_id for mem in retrieved_memories}
            metrics = calculate_metrics(retrieved_memory_ids, ground_truth_memory_ids)
            threshold_metrics_list.append(metrics)
            # --- End repetition ---

        num_queries = len(threshold_metrics_list)
        if num_queries > 0:
            avg_precision = sum(m['precision'] for m in threshold_metrics_list) / num_queries
            avg_recall = sum(m['recall'] for m in threshold_metrics_list) / num_queries
            avg_f1 = sum(m['f1'] for m in threshold_metrics_list) / num_queries
            overall_results[beta] = {"avg_precision": avg_precision, "avg_recall": avg_recall, "avg_f1": avg_f1}
        else:
            overall_results[beta] = {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0}

    print("\n--- KHM Overall Average Results by Beta (Threshold={:.2f}) ---".format(SIMILARITY_THRESHOLD))
    for b, results in overall_results.items():
        print(f"Beta={b:<5.1f}: Avg P={results['avg_precision']:.3f}, Avg R={results['avg_recall']:.3f}, Avg F1={results['avg_f1']:.3f}")

# --- Run with Pytest ---
# Save this file as test_advanced_hopfield.py and run:
# pytest -s test_advanced_hopfield.py

if __name__ == "__main__":
     print("This script is intended to be run with pytest.")
     print("Example: pytest -s test_advanced_hopfield.py")