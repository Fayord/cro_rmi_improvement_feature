from typing import List, Tuple, Callable, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


def find_similar_sentences(
    input_sentence: str,
    candidate_sentences: List[str],
    distance_function: Callable,
    embedding_model: Any,
) -> Tuple[List[str], List[int]]:
    """
    Find the most similar sentences from a list of candidates.
    """
    # Generate embeddings
    input_embedding = embedding_model.get_embedding(input_sentence)
    candidate_embeddings = np.array(
        [embedding_model.get_embedding(sent) for sent in candidate_sentences]
    )

    # Calculate distances
    distances = np.array(
        [
            distance_function(input_embedding, candidate_embedding)
            for candidate_embedding in candidate_embeddings
        ]
    )

    # Sort by distance
    sorted_indices = np.argsort(distances)
    sorted_sentences = [candidate_sentences[i] for i in sorted_indices]

    return sorted_sentences, sorted_indices.tolist()


def find_similar_sentences_batch(
    input_sentences: List[str],
    candidate_sentences: List[str],
    distance_function: Callable[[np.ndarray, np.ndarray], float],
    embedding_model: SentenceTransformer,
) -> List[Dict[str, List]]:
    """
    Find the most similar sentences for multiple input sentences.

    Args:
        input_sentences: List of reference sentences to compare against
        candidate_sentences: List of sentences to compare with
        distance_function: Function to calculate distance between embeddings
        embedding_model: SentenceTransformer model for generating embeddings

    Returns:
        List of dictionaries, each containing:
        - 'similar_sentences': List of most similar sentences
        - 'similar_indices': List of indices of most similar sentences
    """
    results = []
    for input_sentence in input_sentences:
        similar_sentences, similar_indices = find_similar_sentences(
            input_sentence, candidate_sentences, distance_function, embedding_model
        )
        results.append(
            {
                "input_sentence": input_sentence,
                "similar_sentences": similar_sentences,
                "similar_indices": similar_indices,
            }
        )
    return results


def find_similar_sentences_batch_with_embeddings(
    input_sentences: List[str],
    input_embeddings: List[np.ndarray],
    candidate_embeddings: List[np.ndarray],
    candidate_sentences: List[str],
    distance_function: Callable[[np.ndarray, np.ndarray], float],
) -> List[Dict[str, List]]:
    """
    Find the most similar sentences using pre-computed embeddings.

    Args:
        input_sentences: List of input sentences
        input_embeddings: List of pre-computed embeddings for input sentences
        candidate_embeddings: List of pre-computed embeddings for candidate sentences
        candidate_sentences: List of candidate sentences (for output)
        distance_function: Function to calculate distance between embeddings

    Returns:
        List of dictionaries, each containing:
        - 'input_sentence': Original input sentence
        - 'similar_sentences': List of most similar sentences
        - 'similar_indices': List of indices of most similar sentences
    """
    results = []
    for input_sentence, input_embedding in zip(input_sentences, input_embeddings):
        # Calculate distances
        distances = [
            distance_function(input_embedding, candidate_embedding)
            for candidate_embedding in candidate_embeddings
        ]

        # Get sorted indices (ascending order of distances)
        sorted_indices = np.argsort(distances)

        # Get the corresponding sentences
        similar_sentences = [candidate_sentences[idx] for idx in sorted_indices]

        results.append(
            {
                "input_sentence": input_sentence,
                "similar_sentences": similar_sentences,
                "similar_indices": sorted_indices.tolist(),
            }
        )

    return results


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# Example usage:
if __name__ == "__main__":
    # Example distance functions

    # Example usage
    print("START")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("complete load model")

    input_texts = [
        "The risk of market volatility affects investment returns",
        "Cybersecurity threats pose significant risks to organizations",
    ]

    candidates = [
        "Market fluctuations can impact investment performance",
        "The weather is nice today",
        "Financial markets show increased volatility",
        "Data breaches can compromise sensitive information",
        "I like to play tennis",
    ]

    results = find_similar_sentences_batch(
        input_texts, candidates, cosine_distance, model
    )

    print("\nResults for multiple input sentences:")
    for idx, result in enumerate(results):
        print(f"\nInput sentence {idx + 1}: {result["input_sentence"]}")
        print("Most similar sentences (in order):")
        for i, sentence in enumerate(result["similar_sentences"]):
            print(f"{i + 1}. {sentence}")
        print("Corresponding indices:", result["similar_indices"])

    print("\nExample with pre-computed embeddings:")
    input_embeddings = model.encode(input_texts)
    candidate_embeddings = model.encode(candidates)

    results_with_embeddings = find_similar_sentences_batch_with_embeddings(
        input_texts, input_embeddings, candidate_embeddings, candidates, cosine_distance
    )
    print("result with embeding")
    for idx, result in enumerate(results_with_embeddings):
        print(f"\nInput sentence {idx + 1}: {result["input_sentence"]}")
        print("Most similar sentences (in order):")
        for i, sentence in enumerate(result["similar_sentences"]):
            print(f"{i + 1}. {sentence}")
        print("Corresponding indices:", result["similar_indices"])
