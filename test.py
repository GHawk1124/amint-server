# test_hopfield_recall.py
import pytest
import torch
from datetime import datetime
from typing import List, Dict, Set, Tuple

# Assuming your classes are importable from the current directory structure
# If they are in different locations, adjust the import paths accordingly.
try:
    from temporal_hopfield_network import ModernHopfieldNetwork, Memory, DocumentContainer, TextProcessor
    from sqlite import SQLiteManager # Not strictly needed for this test unless loading/saving within test
except ImportError as e:
    print(f"Import Error: {e}. Make sure the test file can access your project modules.")
    # You might need to add the project root to sys.path or configure PYTHONPATH
    # import sys
    # sys.path.append('../') # Example: if test file is in a 'tests' subdirectory
    # from temporal_hopfield_network import ModernHopfieldNetwork, Memory, DocumentContainer, TextProcessor
    # from sqlite import SQLiteManager


# --- Test Configuration ---

# 1. Define Test Documents
DOCUMENTS = {
    "doc1": {
        "title": "Solar System Basics",
        "text": """The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.
Mercury is the closest planet to the Sun. Venus is the second planet.
Earth is the third planet and the only one known to support life. Mars is the fourth planet, often called the Red Planet.
Jupiter is the largest planet, a gas giant. Saturn is known for its prominent rings.
Uranus and Neptune are ice giants in the outer Solar System.""", # Removed trailing newlines for consistency
        "source_type": "article"
    },
    "doc2": {
        "title": "Python Programming",
        "text": """Python is a high-level, interpreted programming language. It emphasizes code readability.
Guido van Rossum began working on Python in the late 1980s.
Its standard library is large and comprehensive. Python supports multiple programming paradigms, including structured, object-oriented and functional programming.
It is often used for web development, data science, artificial intelligence, and scripting.""", # Removed trailing newlines
        "source_type": "reference"
    },
        # Added Document 3
    "doc3": {
        "title": "Apollo 11 Moon Landing",
        "text": """Apollo 11 was the American spaceflight that first landed humans on the Moon.
    Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.
    Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later. Aldrin joined him about 20 minutes later.
    They spent about two and a quarter hours together outside the spacecraft, collecting lunar material. Command module pilot Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface.""",
        "source_type": "historical_summary"
    },
    # Added Document 4
    "doc4": {
        "title": "Photosynthesis Overview",
        "text": """Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. This chemical energy is later released to fuel the organisms' activities.
    This process occurs in chloroplasts, using chlorophyll, the pigment that gives plants their green color. Water and carbon dioxide are used as reactants.
    Oxygen is released as a byproduct. Glucose (a sugar) is produced to store energy.""",
        "source_type": "biology_concept"
    }
}

# Updated QUERIES_GROUND_TRUTH dictionary
QUERIES_GROUND_TRUTH = {
    # Original Queries
    "What is the Solar System?": {"doc1": {0}},
    "Which planets are gas giants?": {"doc1": {2}},
    "Tell me about Python's history": {"doc2": {1}},
    "What is Python used for?": {"doc2": {2}},
    "Does Earth have rings?": {"doc1": {1, 3}},
    # New Queries for Doc 3 (Apollo)
    "Who landed on the Moon in 1969?": {"doc3": {0}}, # Armstrong, Aldrin, Date
    "What did the Apollo 11 astronauts collect?": {"doc3": {2}}, # Collecting lunar material
    "Who was the command module pilot for Apollo 11?": {"doc3": {2}}, # Michael Collins
    # New Queries for Doc 4 (Photosynthesis)
    "What is photosynthesis?": {"doc4": {0}}, # Definition
    "What are the inputs for photosynthesis?": {"doc4": {1}}, # Water, CO2 (reactants)
    "What does photosynthesis release?": {"doc4": {2}}, # Oxygen (byproduct)
    "Where does photosynthesis happen in plants?": {"doc4": {1}} # Chloroplasts
    # Optional broader query
    #"Tell me about space and plants": {"doc1":{0,1,2,3}, "doc3":{0,1,2}, "doc4":{0,1,2}} # Example very broad query
}

# 3. Define Test Parameters
SIMILARITY_THRESHOLDS = [0.65, 0.7, 0.75, 0.8]
CHUNK_SIZE = 2
MAX_CHUNK_LENGTH = 1000

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def text_processor() -> TextProcessor:
    """Initialize the TextProcessor once per module."""
    print("\nInitializing Text Processor...")
    # Ensure NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        import nltk
        nltk.download('punkt', quiet=True)
    return TextProcessor()

@pytest.fixture(scope="module")
def setup_network(text_processor: TextProcessor) -> Tuple[ModernHopfieldNetwork, Dict[str, Dict[int, str]]]:
    """
    Initialize the network, process documents, and store memories.
    Returns the network and a mapping: {doc_id: {chunk_index: memory_id}}
    """
    print("\nSetting up Hopfield Network for testing...")
    # Determine embedding dimension from the model
    try:
        embedding_dim = text_processor.embedding_model.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"Error getting embedding dimension: {e}. Defaulting to 384.")
        embedding_dim = 384 # Default for all-MiniLM-L6-v2 if model loading failed early

    network = ModernHopfieldNetwork(embedding_dim=embedding_dim)
    memory_id_map: Dict[str, Dict[int, str]] = {} # {doc_id: {chunk_idx: mem_id}}

    for doc_id, doc_data in DOCUMENTS.items():
        print(f"Processing document: {doc_id} - {doc_data['title']}")
        # Use the process_document method which adds chunk_index metadata
        document_container, memories = text_processor.process_document(
            title=doc_data['title'],
            text=doc_data['text'],
            chunk_size=CHUNK_SIZE,
            max_chunk_length=MAX_CHUNK_LENGTH,
            use_section_detection=False, # Keep it simple for this example
            source_type=doc_data['source_type'],
            metadata={"original_doc_id": doc_id} # Add original doc ID to container meta if needed
        )
        network.add_document_container(document_container)
        network.batch_store(memories)

        # Store memory IDs for ground truth mapping using chunk_index from metadata
        memory_id_map[doc_id] = {}
        for i, mem in enumerate(memories):
             # process_document adds 'chunk_index' to metadata
             chunk_index = mem.metadata.get("chunk_index")
             if chunk_index is None:
                 print(f"[Warning] Memory {mem.memory_id} for doc {doc_id} missing 'chunk_index' metadata. Using list index {i} as fallback.")
                 chunk_index = i # Fallback, might cause issues if order isn't guaranteed

             memory_id_map[doc_id][chunk_index] = mem.memory_id
             # print(f"  Stored chunk {chunk_index} for {doc_id} with memory_id {mem.memory_id}")


    print(f"Setup complete. Network has {len(network.memories)} memories.")
    # print(f"Memory ID Map: {memory_id_map}") # Uncomment for debugging map creation
    return network, memory_id_map

# --- Helper Function for Metrics ---

def calculate_metrics(retrieved_ids: Set[str], relevant_ids: Set[str]) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1 Score."""
    tp = len(retrieved_ids.intersection(relevant_ids))
    fp = len(retrieved_ids.difference(relevant_ids))
    fn = len(relevant_ids.difference(retrieved_ids))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # If there are no relevant docs, precision is 0 if anything was retrieved, 1 otherwise (or undefined)
    # Let's refine: If relevant_ids is empty, precision is 1.0 if retrieved_ids is also empty, else 0.0
    if not relevant_ids:
        precision = 1.0 if not retrieved_ids else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # If relevant_ids is empty, recall is 1.0 (as there were no relevant items to miss)
    if not relevant_ids:
        recall = 1.0


    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    # If relevant_ids is empty, F1 should follow suit (1.0 if perfect match, 0.0 otherwise)
    if not relevant_ids:
         f1 = 1.0 if not retrieved_ids else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp), # Cast to float for consistency if needed later
        "fp": float(fp),
        "fn": float(fn)
    }


# --- Test Function ---

# Use parametrize to run the test for each query and threshold
@pytest.mark.parametrize("query", QUERIES_GROUND_TRUTH.keys())
@pytest.mark.parametrize("threshold", SIMILARITY_THRESHOLDS)
def test_retrieval_metrics(
    query: str,
    threshold: float,
    setup_network: Tuple[ModernHopfieldNetwork, Dict[str, Dict[int, str]]],
    text_processor: TextProcessor
):
    """
    Tests the retrieval performance for a given query and threshold.
    """
    network, memory_id_map = setup_network
    print(f"\n--- Testing Query: '{query}' | Threshold: {threshold} ---")

    # 1. Get Ground Truth Memory IDs for this query
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

    if not ground_truth_memory_ids and relevant_chunk_indices_by_doc:
         print("[Warning] No ground truth memory IDs were resolved for this query, though mapping exists.")


    # 2. Perform Retrieval
    query_embedding = text_processor.embed_text(query)
    retrieved_memories = network.retrieve(query_embedding, similarity_threshold=threshold)
    retrieved_memory_ids = {mem.memory_id for mem in retrieved_memories}

    # 3. Calculate Metrics
    metrics = calculate_metrics(retrieved_memory_ids, ground_truth_memory_ids)

    # 4. Output / Assertions
    print(f"  Relevant IDs (Expected): {ground_truth_memory_ids or '{}'}")
    print(f"  Retrieved IDs (Actual):  {retrieved_memory_ids or '{}'}")
    print(f"  Metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f} (TP={int(metrics['tp'])}, FP={int(metrics['fp'])}, FN={int(metrics['fn'])})")

    # Example assertion (optional):
    # assert metrics['f1'] >= 0.0, "F1 score should be non-negative"


# --- Optional: Test for Overall Average Performance ---

def test_average_performance(
    setup_network: Tuple[ModernHopfieldNetwork, Dict[str, Dict[int, str]]],
    text_processor: TextProcessor
):
    """Calculates average metrics across all queries for each threshold."""
    network, memory_id_map = setup_network
    print("\n--- Testing Average Performance Across Thresholds ---")

    overall_results: Dict[float, Dict[str, float]] = {} # Store avg results per threshold

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_metrics_list = [] # List to store metrics dict for each query at this threshold
        for query in QUERIES_GROUND_TRUTH.keys():
            # --- (Repeat steps 1-3 from test_retrieval_metrics for each query) ---
            relevant_chunk_indices_by_doc = QUERIES_GROUND_TRUTH.get(query, {})
            ground_truth_memory_ids: Set[str] = set()
            for doc_id, relevant_indices in relevant_chunk_indices_by_doc.items():
                 if doc_id in memory_id_map:
                    doc_map = memory_id_map[doc_id]
                    for chunk_idx in relevant_indices:
                         if chunk_idx in doc_map:
                             ground_truth_memory_ids.add(doc_map[chunk_idx])
            # --- (End of ground truth mapping) ---

            query_embedding = text_processor.embed_text(query)
            retrieved_memories = network.retrieve(query_embedding, similarity_threshold=threshold)
            retrieved_memory_ids = {mem.memory_id for mem in retrieved_memories}
            metrics = calculate_metrics(retrieved_memory_ids, ground_truth_memory_ids)
            threshold_metrics_list.append(metrics)
        # --- (End of loop through queries) ---

        # Calculate average metrics for this threshold
        num_queries = len(threshold_metrics_list)
        if num_queries > 0:
            avg_precision = sum(m['precision'] for m in threshold_metrics_list) / num_queries
            avg_recall = sum(m['recall'] for m in threshold_metrics_list) / num_queries
            avg_f1 = sum(m['f1'] for m in threshold_metrics_list) / num_queries
            overall_results[threshold] = {"avg_precision": avg_precision, "avg_recall": avg_recall, "avg_f1": avg_f1}
        else:
             overall_results[threshold] = {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0}

    # --- (End of loop through thresholds) ---

    print("\n--- Overall Average Results by Threshold ---")
    for thr, results in overall_results.items():
         print(f"Threshold={thr:.2f}: Avg P={results['avg_precision']:.3f}, Avg R={results['avg_recall']:.3f}, Avg F1={results['avg_f1']:.3f}")

# Allow running the script directly if needed for simple checks (though pytest is preferred)
if __name__ == "__main__":
     print("This script is intended to be run with pytest.")
     print("Example: pytest -s test_hopfield_recall.py")
     # You could add basic execution here for debugging fixtures if needed
     # processor = text_processor()
     # setup_network(processor)